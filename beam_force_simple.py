import Sofa
import numpy as np
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(THIS_DIR, "mesh")

carotids_path        = os.path.join(MESH_DIR, "carotids.stl")
aneurysm_collis_path = os.path.join(MESH_DIR, "aneurysm3d_collis.obj")
aneurysm_surf_path   = os.path.join(MESH_DIR, "aneurysm3d_surface.obj")
cava_path            = os.path.join(MESH_DIR, "CavaVeinAndHeart.obj")
key_tip_path         = os.path.join(MESH_DIR, "key_tip.obj")
phantom_path         = os.path.join(MESH_DIR, "phantom.obj")
class PrintTipPos(Sofa.Core.Controller):
    def __init__(self, collision_mo, every=20, **kwargs):
        super().__init__(**kwargs)
        self.collision_mo = collision_mo
        self.every = int(every)
        self.step = 0

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.step % self.every != 0:
            return
        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        if pos.shape[0] > 0:
            print("[TipPos]", pos[-1])
class TipPointUpdater(Sofa.Core.Controller):
    def __init__(self, collision_mo, tip_mo, **kwargs):
        super().__init__(**kwargs)
        self.collision_mo = collision_mo
        self.tip_mo = tip_mo

    def onAnimateBeginEvent(self, e):
        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        if pos.shape[0] == 0:
            return
        self.tip_mo.position.value = [pos[-1].tolist()]  # single point at tip

class TipForceVsGeometryAngle(Sofa.Core.Controller):
    """
    Minimal logger:
      - Computes resultant force over last k tip points (from constraint forces)
      - Computes circumferential angle of lateral force component about tip tangent (theta_force)
      - Computes circumferential angle from closest contact vessel point p2 (theta_geo)
      - Prints magnitude + both angles + their wrapped difference

    Units follow your scene (likely mN if length=mm and time=s).
    """

    def __init__(self,
                 collision_mo,
                 constraint_solver,
                 contact_listener=None,
                 k=1,
                 sample_every=10,
                 eps_force=1e-9,
                 eps_axis=1e-9,
                 **kwargs):
        super().__init__(**kwargs)
        self.collision_mo = collision_mo
        self.solver = constraint_solver
        self.contact_listener = contact_listener

        self.k = int(k)
        self.sample_every = int(sample_every)
        self.eps_force = float(eps_force)
        self.eps_axis = float(eps_axis)

        self.step = 0

    # ----------------- small utilities -----------------

    def _wrap_deg(self, a):
        a = float(a) % 360.0
        return a + 360.0 if a < 0 else a

    def _angdiff_deg(self, a, b):
        # smallest signed difference a-b in (-180,180]
        return (float(a) - float(b) + 180.0) % 360.0 - 180.0

    def _unit(self, v, eps=1e-12):
        n = float(np.linalg.norm(v))
        return None if n < eps else (v / n)

    def _v3_to_np(self, v):
        if v is None:
            return None
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            pass
        try:
            vv = list(v)
            if len(vv) >= 3:
                return np.array([float(vv[0]), float(vv[1]), float(vv[2])], dtype=float)
        except Exception:
            pass
        try:
            return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
        except Exception:
            pass
        try:
            a = np.asarray(v, dtype=float).ravel()
            return a[:3].astype(float) if a.size >= 3 else None
        except Exception:
            return None

    # ----------------- tip frame + angles -----------------

    def _compute_tip_frame(self, pos):
        if pos.shape[0] < 2:
            return None

        p_tip = pos[-1]
        p_prev = pos[-2]
        t_hat = self._unit(p_tip - p_prev)
        if t_hat is None:
            return None

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(up, t_hat)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        e1 = self._unit(np.cross(t_hat, up))
        if e1 is None:
            return None
        e2 = np.cross(t_hat, e1)  # right-handed

        return p_tip, t_hat, e1, e2

    def _force_theta(self, F, t_hat, e1, e2):
        # lateral component about tip axis
        F_perp = F - np.dot(F, t_hat) * t_hat
        Fp = float(np.linalg.norm(F_perp))
        if Fp < self.eps_force:
            return np.nan, 0.0

        x = float(np.dot(F_perp, e1))
        y = float(np.dot(F_perp, e2))
        theta = self._wrap_deg(np.degrees(np.arctan2(y, x)))
        return theta, Fp

    def _theta_from_cp2(self, p_tip, t_hat, e1, e2, p2):
        # project vessel point into cross-section plane and compute theta
        d = p2 - p_tip
        d_perp = d - np.dot(d, t_hat) * t_hat
        n = float(np.linalg.norm(d_perp))
        if n < self.eps_axis:
            return np.nan  # ill-conditioned: contact nearly on axis direction

        r_hat = d_perp / n
        x = float(np.dot(r_hat, e1))
        y = float(np.dot(r_hat, e2))
        return self._wrap_deg(np.degrees(np.arctan2(y, x)))

    # ----------------- contact + force extraction -----------------

    def _n_contacts(self):
        cl = self.contact_listener
        if cl is None:
            return 0
        try:
            return int(cl.getNumberOfContacts())
        except Exception:
            return 0

    def _closest_contact_p2(self):
        """
        Returns (gap_mm, p2_np) for the closest contact (min distance), else None.
        """
        cl = self.contact_listener
        if cl is None:
            return None

        try:
            nC = int(cl.getNumberOfContacts())
        except Exception:
            return None
        if nC <= 0:
            return None

        try:
            dists = np.asarray(cl.getDistances(), dtype=float).ravel()
            if dists.size == 0:
                return None
            i = int(np.nanargmin(dists))
            gap = float(dists[i])
        except Exception:
            return None

        try:
            cps = cl.getContactPoints()
            if cps is None or len(cps) <= i:
                return (gap, None)

            item = cps[i]
            if not (isinstance(item, (list, tuple)) and len(item) >= 4):
                return (gap, None)

            p2 = self._v3_to_np(item[3])
            return (gap, p2)
        except Exception:
            return (gap, None)

    def _tip_force_resultant(self, dt):
        """
        Returns (Fvec, |F|) over last k collision points using:
          f = (J^T * lambda) / dt
        """
        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        npts = pos.shape[0]
        if npts == 0:
            return None

        tip_start = max(0, npts - self.k)

        J = self.collision_mo.constraint.value
        if J is None or getattr(J, "shape", (0, 0))[0] == 0:
            return None

        lambdas = np.asarray(self.solver.constraintForces.value, dtype=float).ravel()
        if lambdas.size == 0:
            return None

        m = min(J.shape[0], lambdas.size)
        if m == 0:
            return None

        f = (J[:m, :].T @ lambdas[:m]) / dt
        if f.size < 3 * npts:
            return None

        Fvec = np.zeros(3, dtype=float)
        for pid in range(tip_start, npts):
            base = 3 * pid
            Fvec += np.array([float(f[base]), float(f[base + 1]), float(f[base + 2])], dtype=float)

        Fmag = float(np.linalg.norm(Fvec))
        return Fvec, Fmag

    # ----------------- SOFA event hook -----------------

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.sample_every > 1 and (self.step % self.sample_every != 0):
            return

        ctx = self.getContext()
        t = float(ctx.time.value)
        dt = float(ctx.dt.value)
        if dt <= 0:
            return

        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        frame = self._compute_tip_frame(pos)
        if frame is None:
            return
        p_tip, t_hat, e1, e2 = frame

        force_out = self._tip_force_resultant(dt)
        if force_out is None:
            return
        Fvec, Fmag = force_out

        theta_force, Fperp = self._force_theta(Fvec, t_hat, e1, e2)

        nC = self._n_contacts()
        gap = np.nan
        theta_geo = np.nan

        cc = self._closest_contact_p2()
        if cc is not None:
            gap, p2 = cc
            if p2 is not None:
                theta_geo = self._theta_from_cp2(p_tip, t_hat, e1, e2, p2)

        theta_diff = np.nan
        if np.isfinite(theta_force) and np.isfinite(theta_geo):
            theta_diff = self._angdiff_deg(theta_force, theta_geo)

        print(
            f"[TipCompare] t={t:.3f} "
            f"|F|={Fmag:.6g}  Fperp={Fperp:.6g}  "
            f"theta_force={theta_force:.2f}deg  theta_geo={theta_geo:.2f}deg  "
            f"dtheta={theta_diff:.2f}deg  "
            f"gap={gap:.6g}mm  nContacts={nC}"
        )
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

import numpy as np
import Sofa

import Sofa
import numpy as np

import Sofa
import numpy as np

import Sofa
import numpy as np

class TipContactForceAndPointRobust(Sofa.Core.Controller):
    def __init__(self,
                 collision_mo,
                 constraint_solver,
                 contact_listener=None,
                 tip_radius=2.0,
                 tip_window=10,
                 sample_every=10,
                 eps=1e-12,
                 gap_contact_gate=1e-4,   # <-- IMPORTANT: allow tiny positive gaps
                 r_perp_min=0.005,        # <-- IMPORTANT: match your observed p2_r ~0.01-0.015
                 force_min=1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.collision_mo = collision_mo
        self.solver = constraint_solver
        self.cl = contact_listener

        self.tip_radius = float(tip_radius)
        self.tip_window = int(tip_window)
        self.sample_every = int(sample_every)
        self.eps = float(eps)

        self.gap_contact_gate = float(gap_contact_gate)
        self.r_perp_min = float(r_perp_min)
        self.force_min = float(force_min)

        self.step = 0
        self._dumped = False

    def _unit(self, v):
        n = float(np.linalg.norm(v))
        return None if n < self.eps else (v / n)

    def _wrap_deg(self, a):
        a = float(a) % 360.0
        return a + 360.0 if a < 0 else a

    def _v3_to_np(self, v):
        if v is None:
            return None
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            pass
        try:
            return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
        except Exception:
            pass
        try:
            a = np.asarray(v, dtype=float).ravel()
            return a[:3].astype(float) if a.size >= 3 else None
        except Exception:
            return None

    def _compute_tip_frame(self, pos):
        if pos.shape[0] < 2:
            return None
        p_tip = pos[-1]
        p_prev = pos[-2]
        t_hat = self._unit(p_tip - p_prev)
        if t_hat is None:
            return None

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(up, t_hat))) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        e1 = self._unit(np.cross(t_hat, up))
        if e1 is None:
            return None
        e2 = np.cross(t_hat, e1)
        return p_tip, t_hat, e1, e2

    def _all_point_forces(self, dt):
        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        npts = pos.shape[0]
        if npts == 0:
            return None

        J = self.collision_mo.constraint.value
        if J is None or getattr(J, "shape", (0, 0))[0] == 0:
            return None

        lambdas = np.asarray(self.solver.constraintForces.value, dtype=float).ravel()
        if lambdas.size == 0:
            return None

        m = min(J.shape[0], lambdas.size)
        if m <= 0:
            return None

        f = (J[:m, :].T @ lambdas[:m]) / dt
        if f.size < 3 * npts:
            return None

        return f[:3 * npts].reshape((npts, 3))

    def _closest_contact_gap_p2(self):
        cl = self.cl
        if cl is None:
            return (np.nan, None, 0)

        try:
            nC = int(cl.getNumberOfContacts())
        except Exception:
            return (np.nan, None, 0)

        if nC <= 0:
            return (np.nan, None, 0)

        try:
            dists = np.asarray(cl.getDistances(), dtype=float).ravel()
            i = int(np.nanargmin(dists)) if dists.size else 0
            gap = float(dists[i]) if dists.size else np.nan
        except Exception:
            i = 0
            gap = np.nan

        try:
            cps = cl.getContactPoints()
        except Exception:
            cps = None

        if cps is None or i >= len(cps):
            return (gap, None, nC)

        item = cps[i]

        if (not self._dumped):
            self._dumped = True
            print("========== CONTACT_DUMP (one-time) ==========")
            print("[listener]", cl.getPathName())
            print("[nC]", nC)
            print("[type(item)]", type(item), " len=", len(item) if hasattr(item, "__len__") else "NA")
            print("[repr(item)]", repr(item))

        # Your observed format: (id1, p1, id2, p2)
        p2 = None
        if isinstance(item, (list, tuple)) and len(item) == 4 and isinstance(item[0], int) and isinstance(item[2], int):
            p2 = self._v3_to_np(item[3])
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            p2 = self._v3_to_np(item[2])

        return (gap, p2, nC)

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.sample_every > 1 and (self.step % self.sample_every != 0):
            return

        ctx = self.getContext()
        t = float(ctx.time.value)
        dt = float(ctx.dt.value)
        if dt <= 0:
            return

        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        if pos.shape[0] < 2:
            return

        frame = self._compute_tip_frame(pos)
        if frame is None:
            return
        p_tip, t_hat, e1, e2 = frame

        # gap/p2
        gap, p2, nC = self._closest_contact_gap_p2()

        # forces
        Fmag_total = np.nan
        F_tip_total = None
        f_point = self._all_point_forces(dt)
        if f_point is not None:
            tip0 = max(0, pos.shape[0] - self.tip_window)
            F_tip_total = np.sum(f_point[tip0:, :], axis=0)
            Fmag_total = float(np.linalg.norm(F_tip_total))

        # infer contact direction
        p_surface = None
        theta = np.nan
        r_p2 = np.nan
        s_p2 = np.nan

        # Use p2 direction when "near contact"
        if p2 is not None and np.isfinite(gap) and gap <= self.gap_contact_gate:
            d = p2 - p_tip
            s_p2 = float(np.dot(d, t_hat))
            d_perp = d - s_p2 * t_hat
            r_p2 = float(np.linalg.norm(d_perp))
            u = self._unit(d_perp)

            if u is not None and r_p2 >= self.r_perp_min:
                p_surface = p_tip + self.tip_radius * u
                theta = self._wrap_deg(np.degrees(np.arctan2(float(np.dot(u, e2)), float(np.dot(u, e1)))))

        # fallback: force lateral direction
        if p_surface is None and F_tip_total is not None and np.isfinite(Fmag_total) and Fmag_total >= self.force_min:
            F_perp = F_tip_total - float(np.dot(F_tip_total, t_hat)) * t_hat
            uF = self._unit(F_perp)
            if uF is not None:
                p_surface = p_tip + self.tip_radius * uF
                theta = self._wrap_deg(np.degrees(np.arctan2(float(np.dot(uF, e2)), float(np.dot(uF, e1)))))

        print(
            f"[TipContact] t={t:.3f} nC={nC} gap={gap:.6g}mm "
            f"|F_tip_total|={Fmag_total:.6g} "
            f"theta={theta:.2f}deg "
            f"p_tip={np.round(p_tip,3)} "
            f"p_surface={None if p_surface is None else np.round(p_surface,3)} "
            f"(p2_r={r_p2:.5g} p2_s={s_p2:.5g})"
        )



def createScene(rootNode):

    rootNode.addObject('RequiredPlugin', name="plug1", pluginName='BeamAdapter Sofa.Component.Constraint.Projective Sofa.Component.LinearSolver.Direct Sofa.Component.ODESolver.Backward Sofa.Component.StateContainer Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Grid Sofa.Component.Visual Sofa.Component.SolidMechanics.Spring Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name="plug2", pluginName='Sofa.Component.AnimationLoop Sofa.Component.Collision.Detection.Algorithm Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Geometry Sofa.Component.Collision.Response.Contact Sofa.Component.Constraint.Lagrangian.Correction Sofa.Component.Constraint.Lagrangian.Solver Sofa.Component.IO.Mesh')
    rootNode.addObject('RequiredPlugin', pluginName='Sofa.Component.Topology.Mapping Sofa.Component.Mapping.Linear Sofa.GL.Component.Rendering3D')

    rootNode.addObject("VisualStyle", displayFlags="showVisualModels hideBehaviorModels showCollisionModels")
    rootNode.findData("bbox").value = "-100 -100 -50 100 100 500"



    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.addObject('GenericConstraintSolver',
                       name='GCS',
                       maxIt=1000, tolerance=1e-6,
                       computeConstraintForces=True)


    catheter_radius = 2
    catheter_E = 2000
    catheter_rho = 1.1e-6

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
                        length=980.0, radius=catheter_radius,
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

    BeamMechanics.addObject(
        'InterventionalRadiologyController',
        name='DeployController',
        template='Rigid3d',
        instruments='BeamInterpolation',
        topology='@MeshLines',
        startingPos=[0, 0, 0, 0, 0, 0, 1],
        xtip=[0],
        printLog=True,
        rotationInstrument=0,
        step=1., speed=2.,
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

    box = add_static_box_rigid(
        rootNode, "CalibrationBox",
        size_x=50.0, size_y=50.0, thickness=5.0,   # make it thick
        translation=(0, 0, 10),
        rotation=(0, 0, 0),
        collision=True,
        visual=True
    )

    vessel_tris = box.getChild('Geo').getObject('triColl')





    cath_points = BeamCollis.getObject('cathPoints')
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
    BeamCollis.addObject(TipContactForceAndPointRobust(
        collision_mo=collision_mo,
        constraint_solver=gcs,
        contact_listener=contact_listener,
        tip_radius=catheter_radius,  # IMPORTANT
        tip_window=10,
        sample_every=10
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