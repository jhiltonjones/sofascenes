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

import Sofa
import numpy as np

class ContactTipCircLogger(Sofa.Core.Controller):
    def __init__(self,
                 contact_listener,
                 collision_mo,            # REQUIRED: catheter collision MechanicalObject (Vec3d)
                 R=0.002,                 # catheter radius [m]
                 every=1,
                 gate=np.inf,
                 r_perp_min=1e-5,
                 show_p1=True,
                 marker_mo=None,          # optional marker MechanicalObject to move to p_circ
                 **kwargs):
        super().__init__(**kwargs)
        self.cl = contact_listener
        self.collision_mo = collision_mo
        self.R = float(R)
        self.every = int(every)
        self.gate = float(gate)
        self.r_perp_min = float(r_perp_min)
        self.show_p1 = bool(show_p1)
        self.marker_mo = marker_mo
        self.step = 0
        self.prev_t_hat = None
        self.prev_e1 = None
        self.prev_r_hat = None
        self.csv_path = "/home/jack/sofascenes/contact_angle5.csv"
        self._csv_initialized = False
        self.prev_theta = None
        self.theta_unwrapped = None




    def _v3(self, v):
        if v is None:
            return None
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            pass
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

        # --- Continuity: keep t_hat direction consistent across time ---
        if self.prev_t_hat is not None and float(np.dot(t_hat, self.prev_t_hat)) < 0.0:
            t_hat = -t_hat  # flip tangent to maintain continuity

        # --- Choose a stable reference axis for building e1 ---
        # Use previous e1 if available; otherwise pick a world axis not parallel to t_hat
        if self.prev_e1 is not None:
            # make e1 the previous e1 projected onto the plane normal to t_hat
            e1 = self.prev_e1 - np.dot(self.prev_e1, t_hat) * t_hat
            e1 = self._unit(e1)
            if e1 is None:
                self.prev_e1 = None  # fallback below
        else:
            e1 = None

        if e1 is None:
            # fallback: pick a world axis that is least aligned with t_hat
            cand_axes = [
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ]
            dots = [abs(float(np.dot(a, t_hat))) for a in cand_axes]
            ref = cand_axes[int(np.argmin(dots))]  # least aligned axis
            e1 = self._unit(np.cross(t_hat, ref))
            if e1 is None:
                return None

        e2 = np.cross(t_hat, e1)

        # --- Continuity: prevent e1 from flipping sign (keeps theta stable) ---
        if self.prev_e1 is not None and float(np.dot(e1, self.prev_e1)) < 0.0:
            e1 = -e1
            e2 = -e2

        # store for next step
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

    def _tip_circumference_point(self, p_tip, t_hat, e1, e2, p_cath, p_env=None):
        v = p_cath - p_tip
        v_perp = v - np.dot(v, t_hat) * t_hat
        n = float(np.linalg.norm(v_perp))

        # fallback if nearly axial: use env->cath direction
        if n < self.r_perp_min and (p_env is not None):
            d = p_cath - p_env
            d_perp = d - np.dot(d, t_hat) * t_hat
            n2 = float(np.linalg.norm(d_perp))
            if n2 >= self.r_perp_min:
                v_perp = d_perp
                n = n2

        if n < 1e-12:
            return None, None, None

        r_hat = v_perp / n



        p_circ = p_tip + self.R * r_hat
        c1 = float(np.dot(r_hat, e1))
        c2 = float(np.dot(r_hat, e2))
        theta = float(np.arctan2(c2, c1))
        if theta < 0:
            theta += 2.0*np.pi
        return p_circ, r_hat, theta


    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.every > 1 and (self.step % self.every != 0):
            return

        t = float(self.getContext().time.value)

        if not self._csv_initialized:
            with open(self.csv_path, "w") as f:
                f.write("t,has_contact,theta_wrapped,theta_unwrapped,gap,d_cath\n")
            self._csv_initialized = True

        # Defaults: no contact (or unusable contact)
        has_contact = 0
        gap = np.nan
        theta = np.nan
        theta_unwrapped = np.nan
        d_cath = np.nan

        # ---- Try compute contact; NEVER return early ----
        try:
            if self.cl is None or self.collision_mo is None:
                raise RuntimeError("missing objects")

            nC = int(self.cl.getNumberOfContacts())
            if nC <= 0:
                raise RuntimeError("no contacts")

            pos = np.asarray(self.collision_mo.position.value, dtype=float)
            frame = self._compute_tip_frame(pos)
            if frame is None:
                raise RuntimeError("no frame")
            p_tip, t_hat, e1, e2 = frame

            dists = np.asarray(self.cl.getDistances(), dtype=float).ravel()
            cps = self.cl.getContactPoints()
            if cps is None or len(cps) == 0:
                raise RuntimeError("no cps")

            idx = 0
            if dists is not None and dists.size == len(cps) and np.any(np.isfinite(dists)):
                idx = int(np.nanargmin(dists))
                gap = float(dists[idx])

            # Gate: treat as "no contact" but STILL log the row
            if np.isfinite(gap) and gap > self.gate:
                raise RuntimeError("gap gated out")

            id1, p1, id2, p2 = self._parse_contact_item(cps[idx])
            if p1 is None or p2 is None:
                raise RuntimeError("bad p1/p2")

            d1 = float(np.linalg.norm(p1 - p_tip))
            d2 = float(np.linalg.norm(p2 - p_tip))
            if d1 <= d2:
                p_cath, p_env = p1, p2
                d_cath = d1
            else:
                p_cath, p_env = p2, p1
                d_cath = d2

            p_circ, r_hat, theta = self._tip_circumference_point(p_tip, t_hat, e1, e2, p_cath, p_env=p_env)
            if theta is None:
                raise RuntimeError("theta None")

            # Optional: choose theta vs theta+pi closest to previous
            def _wrap_pi(a):
                return (a + np.pi) % (2*np.pi) - np.pi

            theta = float(theta)  # ensure plain python float

            if self.prev_theta is not None:
                theta_alt = (theta + np.pi) % (2*np.pi)
                d0 = abs(_wrap_pi(theta - self.prev_theta))
                d1 = abs(_wrap_pi(theta_alt - self.prev_theta))
                if d1 < d0:
                    theta = theta_alt

            # Unwrap
            if self.prev_theta is None:
                self.theta_unwrapped = theta
            else:
                d = _wrap_pi(theta - self.prev_theta)
                self.theta_unwrapped = float(self.theta_unwrapped) + float(d)

            self.prev_theta = theta
            theta_unwrapped = float(self.theta_unwrapped)

            has_contact = 1

        except Exception:
            pass

        # ---- Write one row per step; write NaNs safely ----
        def _fmt(x):
            # prints "nan" for NaN, otherwise numeric with reasonable precision
            if x is None:
                return "nan"
            try:
                if np.isnan(x):
                    return "nan"
            except Exception:
                pass
            return f"{float(x):.10g}"

        with open(self.csv_path, "a") as f:
            f.write(
                f"{t:.6f},{has_contact},{_fmt(theta)},{_fmt(theta_unwrapped)},{_fmt(gap)},{_fmt(d_cath)}\n"
            )



import os
import numpy as np
import Sofa

class CurvatureTorsionLogger(Sofa.Core.Controller):
    def __init__(self, dofs_mo, every=1, csv_path="curv_tau.csv", **kwargs):
        super().__init__(**kwargs)
        self.dofs = dofs_mo
        self.every = int(every)
        self.csv_path = str(csv_path)
        self.step = 0

        # Create file immediately (so you can confirm path/permissions)
        out_dir = os.path.dirname(self.csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(self.csv_path, "w") as f:
            f.write("t,s,kappa,tau\n")

        print(f"[CurvTau] logging to: {self.csv_path}")

    @staticmethod
    def _arc_length(P):
        d = np.linalg.norm(P[1:] - P[:-1], axis=1)
        s = np.zeros(P.shape[0], dtype=float)
        s[1:] = np.cumsum(d)
        return s

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

    def _get_positions(self):
        """
        Works with Rigid3d: position is [x,y,z,qx,qy,qz,qw].
        """
        raw = self.dofs.position.value

        # raw might be list-like of size N, each element length 7
        try:
            X = np.asarray(raw, dtype=float)
        except Exception:
            # fallback: build manually
            X = np.array([list(r) for r in raw], dtype=float)

        if X.ndim != 2 or X.shape[1] < 3:
            return None
        return X[:, :3]

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.every > 1 and (self.step % self.every != 0):
            return

        P = self._get_positions()
        if P is None or P.shape[0] < 4:
            return

        t = float(self.getContext().time.value)

        s = self._arc_length(P)
        kappa = self._curvature_3pt(P)
        tau = self._torsion_dihedral(P, s)

        with open(self.csv_path, "a") as f:
            for i in range(P.shape[0]):
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
        csv_path="/home/jack/sofascenes/curv_tau.csv"
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

    box = add_static_box_rigid(
        rootNode, "CalibrationBox",
        size_x=100.0 * MM, size_y=100.0 * MM, thickness=5.0 * MM,
        translation=(0.0, 0.0, 25.0 * MM),
        rotation=(50, 0, 100),
        collision=True,
        visual=True
    )


    vessel_tris = box.getChild('Geo').getObject('triColl')
    # carotids = add_static_mesh(
    #     rootNode, "Carotids",
    #     carotids_path,
    #     translation=(-1*MM, 4, 0),
    #     rotation=(30, -90, 90),
    #     scale=3.0*MM,
    #     visual=True,
    #     collision=True,     
    #     triangulate=True
    # )
    # vessel_tris = carotids.getObject('triColl')
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
        gate=1e-3,
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