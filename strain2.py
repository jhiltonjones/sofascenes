import Sofa
import numpy as np
import os

catheter_radius = 2.0       


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(THIS_DIR, "mesh")

carotids_path        = os.path.join(MESH_DIR, "carotids.stl")
aneurysm_collis_path = os.path.join(MESH_DIR, "aneurysm3d_collis.obj")
aneurysm_surf_path   = os.path.join(MESH_DIR, "aneurysm3d_surface.obj")
cava_path            = os.path.join(MESH_DIR, "CavaVeinAndHeart.obj")
key_tip_path         = os.path.join(MESH_DIR, "key_tip.obj")
phantom_path         = os.path.join(MESH_DIR, "phantom.obj")

import numpy as np
import os
import Sofa

class ContactTipCircLogger(Sofa.Core.Controller):
    def __init__(self, contact_listener, collision_mo, R,
                 every=1, gate=np.inf, tip_window=2.0,
                 r_perp_min=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.cl = contact_listener
        self.collision_mo = collision_mo
        self.R = float(R)                
        self.every = int(every)
        self.gate = float(gate)    
        self.tip_window = float(tip_window)
        self.r_perp_min = float(r_perp_min)

        self.step = 0
        self.prev_t_hat = None
        self.prev_e1 = None
        self.prev_theta = None
        self.theta_unwrapped = None

        self.csv_path = "/home/jack/sofascenes/tip_contacts_all.csv"
        self._csv_initialized = False

    def _v3(self, v):
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            try:
                return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
            except Exception:
                return None

    def _unit(self, v, eps=1e-12):
        n = float(np.linalg.norm(v))
        return None if n < eps else (v / n)

    def _compute_tip_frame(self, pos):
        if pos.shape[0] < 2:
            return None
        p_tip = pos[-1]
        p_prev = pos[-2]

        t_hat = self._unit(p_tip - p_prev)
        if t_hat is None:
            return None

        # keep t_hat direction consistent
        if self.prev_t_hat is not None and float(np.dot(t_hat, self.prev_t_hat)) < 0.0:
            t_hat = -t_hat

        # build a stable e1
        e1 = None
        if self.prev_e1 is not None:
            e1 = self.prev_e1 - np.dot(self.prev_e1, t_hat) * t_hat
            e1 = self._unit(e1)
            if e1 is None:
                self.prev_e1 = None

        if e1 is None:
            cand_axes = [
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ]
            dots = [abs(float(np.dot(a, t_hat))) for a in cand_axes]
            ref = cand_axes[int(np.argmin(dots))]
            e1 = self._unit(np.cross(t_hat, ref))
            if e1 is None:
                return None

        e2 = np.cross(t_hat, e1)

        # prevent e1 sign flips
        if self.prev_e1 is not None and float(np.dot(e1, self.prev_e1)) < 0.0:
            e1 = -e1
            e2 = -e2

        self.prev_t_hat = t_hat
        self.prev_e1 = e1
        return p_tip, t_hat, e1, e2

    def _parse_contact_item(self, item):
        id1 = id2 = None
        p1 = p2 = None
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            try: id1 = int(item[0])
            except Exception: id1 = None
            p1 = self._v3(item[1])
            try: id2 = int(item[2])
            except Exception: id2 = None
            p2 = self._v3(item[3])
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            try: id1 = int(item[0])
            except Exception: id1 = None
            p1 = self._v3(item[1])
            p2 = self._v3(item[2])
        return id1, p1, id2, p2

    def _r_perp_to_axis(self, p, p_tip, t_hat):
        w = p - p_tip
        w_perp = w - np.dot(w, t_hat) * t_hat
        return float(np.linalg.norm(w_perp))

    def _theta_from_env(self, p_tip, t_hat, e1, e2, p_env):
        # direction from centerline tip toward vessel point
        v = p_env - p_tip
        v_perp = v - np.dot(v, t_hat) * t_hat
        n = float(np.linalg.norm(v_perp))
        if n < 1e-12:
            return None
        r_hat = v_perp / n
        c1 = float(np.dot(r_hat, e1))
        c2 = float(np.dot(r_hat, e2))
        theta = float(np.arctan2(c2, c1))
        if theta < 0:
            theta += 2.0*np.pi
        return theta

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.every > 1 and (self.step % self.every != 0):
            return

        t = float(self.getContext().time.value)

        if not self._csv_initialized:
            out_dir = os.path.dirname(self.csv_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.csv_path, "w") as f:
                # one row per contact per step
                f.write("t,contact_index,id1,id2,d_tip,gap_surf,theta\n")
            self._csv_initialized = True

        # If no contacts at all, optionally write nothing (or write a sentinel row)
        if self.cl is None or self.collision_mo is None:
            return

        nC = int(self.cl.getNumberOfContacts())
        if nC <= 0:
            return

        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        frame = self._compute_tip_frame(pos)
        if frame is None:
            return
        p_tip, t_hat, e1, e2 = frame

        cps = self.cl.getContactPoints()
        if cps is None or len(cps) == 0:
            return

        rows = []
        for j, item in enumerate(cps):
            id1, p1, id2, p2 = self._parse_contact_item(item)
            if p1 is None or p2 is None:
                continue

            # Tip locality based on closeness to p_tip
            d1_tip = float(np.linalg.norm(p1 - p_tip))
            d2_tip = float(np.linalg.norm(p2 - p_tip))
            d_tip = min(d1_tip, d2_tip)
            if d_tip > self.tip_window:
                continue

            # Classify which is catheter-side vs environment-side.
            # From your diagnostics: catheter-side point is near axis (small r_perp).
            r1 = self._r_perp_to_axis(p1, p_tip, t_hat)
            r2 = self._r_perp_to_axis(p2, p_tip, t_hat)
            if r1 <= r2:
                p_cath, p_env = p1, p2
            else:
                p_cath, p_env = p2, p1

            # Surface gap estimate (since getDistances() is NaN)
            d_center = float(np.linalg.norm(p_env - p_cath))
            gap_surf = d_center - self.R

            # Gate in terms of surface gap (mm)
            if np.isfinite(self.gate) and gap_surf > self.gate:
                continue

            theta = self._theta_from_env(p_tip, t_hat, e1, e2, p_env)
            if theta is None:
                continue

            rows.append((t, j, id1, id2, d_tip, gap_surf, theta))

        if not rows:
            return

        with open(self.csv_path, "a") as f:
            for (tt, j, id1, id2, d_tip, gap_surf, theta) in rows:
                f.write(f"{tt:.6f},{j},{id1},{id2},{d_tip:.10g},{gap_surf:.10g},{theta:.10g}\n")



class CurvatureTorsionLogger(Sofa.Core.Controller):
    def __init__(self, dofs_mo, every=1,
                 csv_path="curv_tau.csv",
                 spin_csv_path="tip_spin.csv",
                 **kwargs):
        super().__init__(**kwargs)
        self.dofs = dofs_mo
        self.every = int(every)
        self.csv_path = str(csv_path)
        self.spin_csv_path = str(spin_csv_path)
        self.step = 0

        self.prev_q = None
        self.u_ref = None
        self.phi_unwrapped = None

        # init files
        out_dir = os.path.dirname(self.csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(self.csv_path, "w") as f:
            f.write("t,s,kappa,tau\n")

        with open(self.spin_csv_path, "w") as f:
            f.write("t,phi_wrapped,phi_unwrapped\n")


    @staticmethod
    def _curvature_3pt(P, eps=1e-12):
        N = P.shape[0]
        kappa = np.full(N, np.nan, dtype=float)
        for i in range(1, N - 1):
            a = P[i]   - P[i - 1]
            b = P[i+1] - P[i]
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            c = P[i+1] - P[i-1]
            lc = np.linalg.norm(c)
            if la < eps or lb < eps or lc < eps:
                continue
            area2 = np.linalg.norm(np.cross(a, b))  # 2*Area
            if area2 < eps:
                kappa[i] = 0.0
                continue
            kappa[i] = (2.0 * area2) / (la * lb * lc)
        return kappa

    @staticmethod
    def _torsion_dihedral(P, s, eps=1e-12):
        N = P.shape[0]
        tau = np.full(N, np.nan, dtype=float)

        for i in range(1, N - 2):
            p0, p1, p2, p3 = P[i-1], P[i], P[i+1], P[i+2]
            b1 = p1 - p0
            b2 = p2 - p1
            b3 = p3 - p2

            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)

            n1n = np.linalg.norm(n1)
            n2n = np.linalg.norm(n2)
            b2n = np.linalg.norm(b2)
            if n1n < eps or n2n < eps or b2n < eps:
                continue

            n1u = n1 / n1n
            n2u = n2 / n2n
            t2  = b2 / b2n

            x = np.dot(n1u, n2u)
            y = np.dot(t2, np.cross(n1u, n2u))
            angle = np.arctan2(y, x)

            ds = s[i+1] - s[i]
            if ds < eps:
                continue

            tau[i+1] = angle / ds
        return tau


    @staticmethod
    def _arc_length(P):
        d = np.linalg.norm(P[1:] - P[:-1], axis=1)
        s = np.zeros(P.shape[0], dtype=float)
        s[1:] = np.cumsum(d)
        return s
    def _get_positions(self):
        raw = self.dofs.position.value
        # print(f"Raw tip Position [-1]: {raw[-1]}")
        # print(f"Raw tip Position [-2]: {raw[-2]}")
        try:
            X = np.asarray(raw, dtype=float)
        except Exception:
            X = np.array([list(r) for r in raw], dtype=float)

        if X.ndim != 2 or X.shape[1] < 7:
            return None, None
        return X[:, :3], X[:, 3:7]
    def _quat_fix(self, q):
        # enforce q continuity (q and -q are the same rotation)
        if getattr(self, "prev_q", None) is not None and float(np.dot(q, self.prev_q)) < 0.0:
            q = -q
        self.prev_q = q
        return q

    def _quat_to_R(self, q):
        # q = [qx,qy,qz,qw]
        x, y, z, w = [float(v) for v in q]
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
            [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
            [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)],
        ], dtype=float)

    def _signed_angle_about_axis(self, u_ref, u_now, axis, eps=1e-12):
        axis = axis / max(eps, np.linalg.norm(axis))
        u_ref = u_ref - np.dot(u_ref, axis) * axis
        u_now = u_now - np.dot(u_now, axis) * axis
        n1 = np.linalg.norm(u_ref); n2 = np.linalg.norm(u_now)
        if n1 < eps or n2 < eps:
            return None
        u_ref /= n1; u_now /= n2
        x = float(np.dot(u_ref, u_now))
        y = float(np.dot(axis, np.cross(u_ref, u_now)))
        return float(np.arctan2(y, x))
    @staticmethod
    def active_polyline(P, eps_seg=1e-6, min_pts=6):
        P = np.asarray(P, float)
        if P.shape[0] < min_pts:
            return P

        d = np.linalg.norm(P[1:] - P[:-1], axis=1)
        good = d > eps_seg
        if not np.any(good):
            return P[:0]  # empty

        i0 = int(np.argmax(good))  # first True
        P2 = P[i0:]

        return P2 if P2.shape[0] >= min_pts else P2


    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.every > 1 and (self.step % self.every != 0):
            return

        P, Q = self._get_positions()
        if P is None or P.shape[0] < 4:
            return
        P_col = np.asarray(self.getContext().getRoot().getChild("BeamModel")
                        .getChild("CollisionModel").getObject("CollisionDOFs").position.value, dtype=float)
        # if self.step %10==0:
        #     P_dofs, _ = self._get_positions()  # your current source
        #     P_col = np.asarray(self.getContext().getRoot().getChild("BeamModel")
        #                     .getChild("CollisionModel").getObject("CollisionDOFs").position.value, dtype=float)

        #     print("DOFs:", P_dofs.shape, "mid", P_dofs[len(P_dofs)//2], "last", P_dofs[-1])
        #     print("COL :", P_col.shape,  "mid", P_col[len(P_col)//2],  "last", P_col[-1])


        t = float(self.getContext().time.value)

        # axis from centerline (robust)
        t_hat = P[-1] - P[-2]
        n = np.linalg.norm(t_hat)
        if n < 1e-12:
            return
        t_hat = t_hat / n

        # tip rotation matrix from quaternion
        q_tip = self._quat_fix(np.asarray(Q[-1], float))
        Rm = self._quat_to_R(q_tip)

        # choose a "stripe" direction in LOCAL frame (must not be parallel to the beam's local axis)
        # If your beam axis is local X, pick local Y as stripe. If axis is local Z, pick local X, etc.
        u_local = np.array([0.0, 1.0, 0.0])
        u_now = Rm @ u_local

        # project onto cross-section plane
        u_now = u_now - np.dot(u_now, t_hat) * t_hat
        nu = np.linalg.norm(u_now)
        if nu < 1e-12:
            return
        u_now /= nu

        if self.u_ref is None:
            self.u_ref = u_now
            self.phi_unwrapped = 0.0
        else:
            dphi = self._signed_angle_about_axis(self.u_ref, u_now, t_hat)
            if dphi is not None:
                self.phi_unwrapped += dphi
                self.u_ref = u_now
        P_use = self.active_polyline(P_col, eps_seg=1e-4)  # tune eps to your mm scale
        if P_use is None or P_use.shape[0] < 4:
            return

        s = self._arc_length(P_use)
        kappa = self._curvature_3pt(P_use)
        tau = self._torsion_dihedral(P_use, s)

        phi_wrapped = float(self.phi_unwrapped % (2.0*np.pi))
        with open(self.spin_csv_path, "a") as f:
            f.write(f"{t:.6f},{phi_wrapped:.10g},{float(self.phi_unwrapped):.10g}\n")

        with open(self.csv_path, "a") as f:
            for i in range(P_use.shape[0]):

                kv = "nan" if np.isnan(kappa[i]) else f"{kappa[i]:.10g}"
                tv = "nan" if np.isnan(tau[i])   else f"{tau[i]:.10g}"
                f.write(f"{t:.6f},{s[i]:.10g},{kv},{tv}\n")




def add_static_box_rigid(parent, name,
                         size_x=200.0, size_y=200.0, thickness=50.0,
                         translation=(0, 0, 0),
                         rotation=(0, 0, 0),
                         color=(0.7, 0.7, 0.7, 0.4),
                         collision=True,
                         visual=True):
    root = parent.addChild(name)

    # Rigid DOF for pose
    root.addObject('MechanicalObject', name='rigidDOF', template='Rigid3d',
                   position=[[translation[0], translation[1], translation[2], 0, 0, 0, 1]],
                   rotation=list(rotation),
                   showObject=False)

    geo = root.addChild('Geo')

    hx = size_x * 0.5
    hy = size_y * 0.5
    hz = thickness * 0.5

    positions = [
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ]

    triangles = [
        # bottom
        [0, 2, 1], [0, 3, 2],
        # top
        [4, 5, 6], [4, 6, 7],
        # -y
        [0, 1, 5], [0, 5, 4],
        # +y
        [3, 7, 6], [3, 6, 2],
        # -x
        [0, 4, 7], [0, 7, 3],
        # +x
        [1, 2, 6], [1, 6, 5],
    ]

    geo.addObject('MeshTopology', name='topo', position=positions, triangles=triangles)
    geo.addObject('MechanicalObject', name='meshDOF', template='Vec3d',
                  position=positions, showObject=False)
    geo.addObject('RigidMapping', input='@../rigidDOF', output='@meshDOF')

    if collision:
        geo.addObject('TriangleCollisionModel', name='triColl', moving=False, simulated=False)
        geo.addObject('LineCollisionModel', name='lineColl', moving=False, simulated=False)
        geo.addObject('PointCollisionModel', name='ptColl', moving=False, simulated=False)

    if visual:
        v = geo.addChild('Visual')
        v.addObject('OglModel', name='ogl', src='@../topo', color=list(color))
        v.addObject('IdentityMapping', input='@../meshDOF', output='@ogl')

    return root
def add_static_mesh(parent, name, filename,
                    translation=(0,0,0), rotation=(0,0,0), scale=1.0,
                    visual=True, collision=True,
                    flipNormals=False, triangulate=True):
    """
    Adds a static mesh with optional visual + collision models.
    - For collision against a catheter, TriangleCollisionModel is typical.
    - If you only want visuals, set collision=False.
    """
    n = parent.addChild(name)

    # Pick loader based on extension
    ext = filename.split('.')[-1].lower()
    if ext == "stl":
        n.addObject('MeshSTLLoader', name='loader', filename=filename,
                    flipNormals=flipNormals, triangulate=triangulate, rotation=list(rotation))
        # STL loader output typically provides position + triangles
        n.addObject('MeshTopology', name='topo',
                    position='@loader.position',
                    triangles='@loader.triangles')
    elif ext == "obj":
        # MeshOBJLoader provides position + triangles/quads/edges depending on file
        n.addObject('MeshOBJLoader', name='loader', filename=filename,
                    flipNormals=flipNormals, triangulate=triangulate, rotation=list(rotation))
        n.addObject('MeshTopology', name='topo',
                    position='@loader.position',
                    triangles='@loader.triangles')
        # Note: if your OBJ has quads only, triangulate=True is important.
    else:
        raise ValueError(f"Unsupported mesh extension: {ext}")

    # Mechanical state (even for static objects)
    n.addObject('MechanicalObject', name='dofs',
                translation=list(translation), rotation=list(rotation), scale=scale,
                showObject=False, showObjectScale=1.0)

    # IMPORTANT: ensure topology follows the MechanicalObject transform
    # In many scenes this is implicit; if you see transform not applied, add a mapping:
    # n.addObject('IdentityMapping', input='@dofs', output='@topo')  # only if needed

    if collision:
        # Static collision object
        n.addObject('TriangleCollisionModel', name='triColl',
                    moving=False, simulated=False)
        n.addObject('LineCollisionModel', name='lineColl',
                    moving=False, simulated=False)
        n.addObject('PointCollisionModel', name='ptColl',
                    moving=False, simulated=False)

    if visual:
        visu = n.addChild('Visual')
        visu.addObject('OglModel', name='ogl', src='@../loader', color=[1.0, 1.0, 1.0, 0.2])

        visu.addObject('IdentityMapping')  # maps MechanicalObject to OglModel

    return n


def createScene(rootNode):

    rootNode.addObject('RequiredPlugin', name="plug1", pluginName='BeamAdapter Sofa.Component.Constraint.Projective Sofa.Component.LinearSolver.Direct Sofa.Component.ODESolver.Backward Sofa.Component.StateContainer Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Grid Sofa.Component.Visual Sofa.Component.SolidMechanics.Spring Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name="plug2", pluginName='Sofa.Component.AnimationLoop Sofa.Component.Collision.Detection.Algorithm Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Geometry Sofa.Component.Collision.Response.Contact Sofa.Component.Constraint.Lagrangian.Correction Sofa.Component.Constraint.Lagrangian.Solver Sofa.Component.IO.Mesh')
    rootNode.addObject('RequiredPlugin', pluginName='Sofa.Component.Topology.Mapping Sofa.Component.Mapping.Linear Sofa.GL.Component.Rendering3D')
    # rootNode.dt = 0.005   # 5 ms
    rootNode.addObject("VisualStyle", displayFlags="showVisualModels hideBehaviorModels showCollisionModels")


    rootNode.findData('dt').value = 0.005



    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.addObject('GenericConstraintSolver',
                       name='GCS',
                       maxIt=1000, tolerance=1e-6,
                       computeConstraintForces=True)


    # MM = 1e-3  # millimeter in meters
    MM = 1
    # --- Catheter material (SI) ---
    catheter_radius = 2.0 * MM          # 2 mm -> 0.002 m
    catheter_E   = 2.0e6              # 2e9 Pa -> 2e6 kg/(mm*s^2)
    catheter_rho = 1.1e-6             # 1100 kg/m^3 -> 1.1e-6 kg/mm^3

    # --- Catheter geometry (SI) ---
    catheter_length = 980.0 * MM        # 980 mm -> 0.98 m


    rootNode.addObject('CollisionPipeline', draw='0', depth='6', verbose='1')
    rootNode.addObject('BruteForceBroadPhase', name='N2')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('LocalMinDistance',
                    contactDistance=str(catheter_radius),
                    alarmDistance=str(2 * catheter_radius),
                    name='localmindistance',
                    angleCone='0.2')

    rootNode.addObject('CollisionResponse', name='Response', response='FrictionContactConstraint')




    topoLines = rootNode.addChild('EdgeTopology')

    topoLines.addObject('RodStraightSection', name='StraightSection',
                        length=catheter_length, radius=catheter_radius,
                        nbBeams=50, nbEdgesCollis=50, nbEdgesVisu=200,
                        youngModulus=catheter_E, massDensity=catheter_rho, poissonRatio=0.45)


    topoLines.addObject('WireRestShape', name='BeamRestShape', template="Rigid3d",
                        wireMaterials="@StraightSection")

    topoLines.addObject('EdgeSetTopologyContainer', name='meshLines')
    topoLines.addObject('EdgeSetTopologyModifier', name='Modifier')
    topoLines.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo', template='Rigid3d')
    topoLines.addObject('MechanicalObject', name='dofTopo2', template='Rigid3d')

    BeamMechanics = rootNode.addChild('BeamModel')
    BeamMechanics.addObject('EulerImplicitSolver', rayleighStiffness=0.2, rayleighMass=0.1)
    BeamMechanics.addObject('BTDLinearSolver', verification=False, subpartSolve=False, verbose=False)

    BeamMechanics.addObject('RegularGridTopology', name='MeshLines',
                            nx=61, ny=1, nz=1,
                            xmax=0.0, xmin=0.0, ymin=0, ymax=0, zmax=0, zmin=0,
                            p0=[0, 0, 0])

    BeamMechanics.addObject('MechanicalObject', showIndices=False, name='DOFs', template='Rigid3d', ry=-90)
    BeamMechanics.addObject('WireBeamInterpolation', name='BeamInterpolation',
                            WireRestShape='@../EdgeTopology/BeamRestShape', printLog=False)

    BeamMechanics.addObject('AdaptiveBeamForceFieldAndMass', name='BeamForceField',
                            massDensity=catheter_rho, interpolation='@BeamInterpolation')
    dofs_mo = BeamMechanics.getObject('DOFs')

    BeamMechanics.addObject(CurvatureTorsionLogger(
        dofs_mo=dofs_mo,
        every=1,
        csv_path="/home/jack/sofascenes/curv_tau.csv",
        spin_csv_path="/home/jack/sofascenes/tip_spin.csv",
    ))



    BeamMechanics.addObject(
        'InterventionalRadiologyController',
        name='DeployController',
        template='Rigid3d',
        instruments='BeamInterpolation',
        topology='@MeshLines',
        startingPos=[0, 0, 0, 0, 0, 0, 1.0],
        xtip=[0],
        printLog=True,
        rotationInstrument=0,
        step  = 1.0 * MM,      # 1 mm
        speed = 20.0 * MM,      # 2 mm/s        
        listening=True,
        controlledInstrument=0
    )

    BeamMechanics.addObject('LinearSolverConstraintCorrection', wire_optimization='true', printLog=False)
    BeamMechanics.addObject('FixedProjectiveConstraint', indices=0, name='FixedConstraint')
    BeamMechanics.addObject('RestShapeSpringsForceField',
                            points='@DeployController.indexFirstNode',
                            angularStiffness=1e8, stiffness=1e8)

    BeamCollis = BeamMechanics.addChild('CollisionModel')
    BeamCollis.activated = True
    BeamCollis.addObject('EdgeSetTopologyContainer', name='collisEdgeSet')
    BeamCollis.addObject('EdgeSetTopologyModifier', name='colliseEdgeModifier')
    BeamCollis.addObject('MechanicalObject', name='CollisionDOFs')
    BeamCollis.addObject('MultiAdaptiveBeamMapping', controller='../DeployController',
                         useCurvAbs=True, printLog=False, name='collisMap')

    BeamCollis.addObject('LineCollisionModel', name='cathLine', proximity=catheter_radius*0.25)
    BeamCollis.addObject('PointCollisionModel', name='cathPoints', proximity=catheter_radius*0.25)



    # --- References needed by the simplified logger ---
    gcs = rootNode.getObject('GCS')
    collision_mo = BeamCollis.getObject('CollisionDOFs')

    # --- Visual catheter (unchanged) ---
    VisuCath = BeamMechanics.addChild('VisuCatheter')
    VisuCath.addObject('MechanicalObject', name="Quads", template="Vec3d")
    VisuCath.addObject('QuadSetTopologyContainer', name="TubeQuads")
    VisuCath.addObject('QuadSetTopologyModifier', name="TubeQuadsModifier")
    VisuCath.addObject('QuadSetGeometryAlgorithms', name="TubeGeom", template="Vec3d")

    VisuCath.addObject(
        'Edge2QuadTopologicalMapping',
        name="Edge2Quad",
        nbPointsOnEachCircle=12,
        radius=catheter_radius,
        input='@../../EdgeTopology/meshLines',
        output='@TubeQuads',
        flipNormals=True,
        printLog=False
    )

    VisuCath.addObject(
        'AdaptiveBeamMapping',
        name="VisuMapCath",
        useCurvAbs=True,
        isMechanical=False,
        interpolation='@../BeamInterpolation',
        printLog=False
    )

    VisuOgl = VisuCath.addChild('Ogl')
    VisuOgl.addObject('OglModel', name="CatheterVisual", src='@../TubeQuads', color='white')
    VisuOgl.addObject('IdentityMapping', input='@../Quads', output='@CatheterVisual')

    # box = add_static_box_rigid(
    #     rootNode, "CalibrationBox",
    #     size_x=100.0 * MM, size_y=100.0 * MM, thickness=5.0 * MM,
    #     translation=(0.0, 0.0, 25.0 * MM),
    #     rotation=(50, 0, 0),
    #     collision=True,
    #     visual=True
    # )


    # vessel_tris = box.getChild('Geo').getObject('triColl')
    carotids = add_static_mesh(
        rootNode, "Carotids",
        carotids_path,
        translation=(-1*MM, 4, 0),
        rotation=(30, -90, 90),
        scale=3.0*MM,
        visual=True,
        collision=True,     
        triangulate=True
    )
    vessel_tris = carotids.getObject('triColl')
    cath_points = BeamCollis.getObject('cathPoints')


    # cath_points = BeamCollis.getObject('cathPoints')
    cm1 = '@' + cath_points.getPathName()


    cm2 = '@' + vessel_tris.getPathName()

    contact_listener = rootNode.addObject(
        'ContactListener',
        name='CathPlaneContactListener',
        collisionModel1=cm1,
        collisionModel2=cm2,
        listening=True
    )


    # # --- Replace old logger with the simplified one ---
    # BeamCollis.addObject(TipForceVsGeometryAngle(
    #     collision_mo=collision_mo,
    #     constraint_solver=gcs,
    #     contact_listener=contact_listener,
    #     k=1,               # last 1 collision point (your previous setting)
    #     sample_every=1
    # ))
    # BeamCollis.addObject(PrintTipPos(collision_mo=collision_mo, every=20))
    # BeamCollis.addObject(TipContactForceAndPointRobust(
    #     collision_mo=collision_mo,
    #     constraint_solver=gcs,
    #     contact_listener=contact_listener,
    #     tip_radius=catheter_radius,  # IMPORTANT
    #     tip_window=1,
    #     sample_every=1
    # ))


    BeamCollis.addObject(ContactTipCircLogger(
        contact_listener=contact_listener,
        collision_mo=collision_mo,
        R=catheter_radius,
        every=1,
        gate=2,
      
        show_p1=True
    ))





def main():
    import SofaRuntime
    import Sofa.Gui

    root = Sofa.Core.Node('root')
    createScene(root)
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init('myscene', 'qglviewer')
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == '__main__':
    main()