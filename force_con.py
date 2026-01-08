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

        if self.cl is None or self.collision_mo is None:
            return

        try:
            nC = int(self.cl.getNumberOfContacts())
        except Exception:
            return
        if nC <= 0:
            return

        # read collision positions and compute tip frame
        try:
            pos = np.asarray(self.collision_mo.position.value, dtype=float)
        except Exception:
            return
        frame = self._compute_tip_frame(pos)
        if frame is None:
            return
        p_tip, t_hat, e1, e2 = frame

        # distances and contact points
        try:
            dists = np.asarray(self.cl.getDistances(), dtype=float).ravel()
        except Exception:
            dists = None
        try:
            cps = self.cl.getContactPoints()
        except Exception:
            cps = None
        if cps is None or len(cps) == 0:
            return

        idx = 0
        gap = np.nan
        if dists is not None and dists.size == len(cps) and np.any(np.isfinite(dists)):
            idx = int(np.nanargmin(dists))
            gap = float(dists[idx])

        if np.isfinite(gap) and gap > self.gate:
            return

        id1, p1, id2, p2 = self._parse_contact_item(cps[idx])
        if p1 is None or p2 is None:
            return

        # decide which point is catheter-side by distance to p_tip
        d1 = float(np.linalg.norm(p1 - p_tip))
        d2 = float(np.linalg.norm(p2 - p_tip))
        if d1 <= d2:
            p_cath, p_env = p1, p2
            id_cath, id_env = id1, id2
            d_cath, d_env = d1, d2
            cath_label = "p1"
        else:
            p_cath, p_env = p2, p1
            id_cath, id_env = id2, id1
            d_cath, d_env = d2, d1
            cath_label = "p2"

        p_circ, r_hat, theta = self._tip_circumference_point(p_tip, t_hat, e1, e2, p_cath, p_env=p_env)
        if theta is None:
            return
        def _wrap_pi(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        # after theta computed
        if self.prev_theta is not None:
            theta_alt = (theta + np.pi) % (2*np.pi)   # corresponds to flipping r_hat
            d0 = abs(_wrap_pi(theta - self.prev_theta))
            d1 = abs(_wrap_pi(theta_alt - self.prev_theta))
            if d1 < d0:
                theta = theta_alt
                r_hat = -r_hat
                p_circ = p_tip + self.R * r_hat

        t = float(self.getContext().time.value)

        # continuity on r_hat to prevent pi-jumps
        if self.prev_r_hat is not None and float(np.dot(r_hat, self.prev_r_hat)) < 0.0:
            r_hat = -r_hat
        self.prev_r_hat = r_hat


        # primary log
        if p_circ is not None:
            print(f"[TipCirc] t={t:.6f} nC={nC} idx={idx} gap={gap:.3e} "
                  f"cath={cath_label} id_cath={id_cath} d_cath={d_cath:.3e} "
                  f"p_cath={np.array2string(p_cath, precision=6)} "
                  f"p_circ={np.array2string(p_circ, precision=6)} "
                  f"theta(rad)={theta:.6f}")
        else:
            print(f"[TipCirc] t={t:.6f} nC={nC} idx={idx} gap={gap:.3e} "
                  f"cath={cath_label} id_cath={id_cath} d_cath={d_cath:.3e} "
                  f"p_cath={np.array2string(p_cath, precision=6)} "
                  f"(no circumference point: radial ill-conditioned)")

        # optional secondary log
        if self.show_p1:
            print(f"         p1 id1={id1} d(p1,tip)={d1:.3e} p1={np.array2string(p1, precision=6)}")
            print(f"         p2 id2={id2} d(p2,tip)={d2:.3e} p2={np.array2string(p2, precision=6)}")
        if self.step % 50 == 0 and p_circ is not None:
            v = p_circ - p_tip
            print("check |v|=", np.linalg.norm(v), " axial=", abs(np.dot(v, t_hat)))

        # move marker to circumference point (more useful than p2)
        if self.marker_mo is not None and p_circ is not None:
            try:
                self.marker_mo.position.value = [p_circ.tolist()]
            except Exception:
                pass
        if not self._csv_initialized:
            with open(self.csv_path, "w") as f:
                f.write("t,theta_wrapped,theta_unwrapped,gap,d_cath\n")
            self._csv_initialized = True

        # incremental unwrap (or omit and unwrap in post)
        if self.prev_theta is None:
            self.theta_unwrapped = theta
        else:
            d = theta - self.prev_theta
            d = (d + np.pi) % (2*np.pi) - np.pi
            self.theta_unwrapped += d
        self.prev_theta = theta

        with open(self.csv_path, "a") as f:
            f.write(f"{t:.6f},{theta:.10f},{self.theta_unwrapped:.10f},{gap:.6e},{d_cath:.6e}\n")





class TipContactForceAndPointRobust(Sofa.Core.Controller):
    def __init__(self,
                 collision_mo,
                 constraint_solver,
                 contact_listener=None,
                 tip_radius=2.0,
                 tip_window=10,
                 sample_every=10,
                 eps=1e-12,
                 gap_contact_gate=1e-3,   # m
                 r_perp_min=5e-6,         # m
                 force_min=1e-6,
                 debug_shapes_every=50,
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

        self.debug_shapes_every = int(debug_shapes_every)

        self.step = 0
        self._dumped = False
        self._coord_dumped = False

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

    def _all_point_forces_from_constraints(self, dt):
        """
        WARNING: This is NOT guaranteed to be contact-only.
        It is a 'reaction-like' force reconstructed from the constraints
        seen by the solver and the constraint matrix attached to collision_mo.
        """
        pos = np.asarray(self.collision_mo.position.value, dtype=float)
        npts = pos.shape[0]
        if npts == 0:
            return None, None, None

        J = self.collision_mo.constraint.value
        if J is None or getattr(J, "shape", (0, 0))[0] == 0:
            return None, J, None

        lambdas = np.asarray(self.solver.constraintForces.value, dtype=float).ravel()
        if lambdas.size == 0:
            return None, J, lambdas

        m = min(J.shape[0], lambdas.size)
        if m <= 0:
            return None, J, lambdas

        f = (J[:m, :].T @ lambdas[:m]) / dt
        if f.size < 3 * npts:
            return None, J, lambdas

        return f[:3 * npts].reshape((npts, 3)), J, lambdas

    # --- Add these helpers inside your class ---

    def _fmtv(self, v, prec=6):
        return np.array2string(np.asarray(v, dtype=float), precision=prec, suppress_small=False)

    def _closest_contact_gap_p2(self):
        cl = self.cl
        if cl is None:
            return (np.nan, None, None, None, None, 0)

        try:
            nC = int(cl.getNumberOfContacts())
        except Exception:
            return (np.nan, None, None, None, None, 0)

        if nC <= 0:
            return (np.nan, None, None, None, None, 0)

        gap = np.nan
        idx = 0
        dists = None
        try:
            dists = np.asarray(cl.getDistances(), dtype=float).ravel()
            if dists.size > 0 and np.any(np.isfinite(dists)):
                idx = int(np.nanargmin(dists))
                gap = float(dists[idx])
        except Exception:
            pass

        try:
            cps = cl.getContactPoints()
        except Exception:
            cps = None

        if cps is None or idx >= len(cps):
            return (gap, None, None, None, None, nC)

        item = cps[idx]

        if not self._dumped:
            self._dumped = True
            print("========== CONTACT_DUMP (one-time) ==========")
            print("[listener]", cl.getPathName())
            print("[nC]", nC)
            print("[len(distances)]", len(dists) if dists is not None else "NA")
            print("[type(item)]", type(item), " len=", len(item) if hasattr(item, "__len__") else "NA")
            print("[repr(item)]", repr(item))

        # Expected format: (id1, p1, id2, p2)
        id1 = None
        id2 = None
        p1 = None
        p2 = None

        if isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                id1 = int(item[0])
            except Exception:
                id1 = None
            p1 = self._v3_to_np(item[1])

            try:
                id2 = int(item[2])
            except Exception:
                id2 = None
            p2 = self._v3_to_np(item[3])
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            # Fallback format (older/other builds)
            try:
                id1 = int(item[0])
            except Exception:
                id1 = None
            p1 = self._v3_to_np(item[1])
            p2 = self._v3_to_np(item[2])

        return (gap, id1, p1, id2, p2, nC)


    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.every > 1 and (self.step % self.every != 0):
            return

        # breadcrumb every 200 steps
        if self.step % 200 == 0:
            print(f"[TipCirc] alive step={self.step}")

        if self.cl is None or self.collision_mo is None:
            if self.step % 200 == 0:
                print("[TipCirc] missing cl or collision_mo")
            return

        try:
            nC = int(self.cl.getNumberOfContacts())
        except Exception as e:
            if self.step % 200 == 0:
                print("[TipCirc] getNumberOfContacts failed:", e)
            return

        if nC <= 0:
            if self.step % 200 == 0:
                print("[TipCirc] nC=0")
            return

        try:
            pos = np.asarray(self.collision_mo.position.value, dtype=float)
        except Exception as e:
            if self.step % 200 == 0:
                print("[TipCirc] collision_mo.position read failed:", e)
            return

        frame = self._compute_tip_frame(pos)
        if frame is None:
            if self.step % 200 == 0:
                print("[TipCirc] tip frame None (need >=2 points, nonzero tangent)")
            return
        p_tip, t_hat, e1, e2 = frame

        try:
            dists = np.asarray(self.cl.getDistances(), dtype=float).ravel()
        except Exception:
            dists = None
        try:
            cps = self.cl.getContactPoints()
        except Exception:
            cps = None
        if cps is None or len(cps) == 0:
            if self.step % 200 == 0:
                print("[TipCirc] cps empty")
            return

        idx = 0
        gap = np.nan
        if dists is not None and dists.size == len(cps) and np.any(np.isfinite(dists)):
            idx = int(np.nanargmin(dists))
            gap = float(dists[idx])

        if np.isfinite(gap) and gap > self.gate:
            if self.step % 50 == 0:
                print(f"[TipCirc] gated by gap: gap={gap:.3e} > gate={self.gate:.3e}")
            return

        id1, p1, id2, p2 = self._parse_contact_item(cps[idx])
        if p1 is None or p2 is None:
            if self.step % 200 == 0:
                print("[TipCirc] parsed p1/p2 None")
            return

        d1 = float(np.linalg.norm(p1 - p_tip))
        d2 = float(np.linalg.norm(p2 - p_tip))

        if d1 <= d2:
            p_cath, p_env = p1, p2
            id_cath, d_cath, cath_label = id1, d1, "p1"
        else:
            p_cath, p_env = p2, p1
            id_cath, d_cath, cath_label = id2, d2, "p2"

        # --- TIP-ONLY ACCEPTANCE GATE ---
        tip_ok_dist = True
        if self.tip_max_dist is not None:
            tip_ok_dist = (d_cath <= self.tip_max_dist)

        K = max(1, min(self.tip_window_pts, pos.shape[0]))
        tip_block = pos[-K:, :]
        dmin_tipblock = float(np.min(np.linalg.norm(tip_block - p_cath[None, :], axis=1)))
        tip_ok_window = (dmin_tipblock <= 1e-3)

        if not (tip_ok_dist and tip_ok_window):
            if self.step % 50 == 0:
                print(f"[TipCirc] rejected: cath={cath_label} d_cath={d_cath:.3e} "
                    f"tip_ok_dist={tip_ok_dist} tip_ok_window={tip_ok_window} "
                    f"dmin_tipblock={dmin_tipblock:.3e}")
            return

        # compute circumference + print
        p_circ, r_hat, theta = self._tip_circumference_point(p_tip, t_hat, e1, e2, p_cath, p_env=p_env)
        t = float(self.getContext().time.value)

        print(f"[TipCirc] t={t:.6f} nC={nC} gap={gap:.3e} cath={cath_label} id_cath={id_cath} "
            f"d_cath={d_cath:.3e} theta={None if theta is None else f'{theta:.6f}'}")
        if p_circ is not None and self.marker_mo is not None:
            try:
                self.marker_mo.position.value = [p_circ.tolist()]
            except Exception:
                pass
            





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


def createScene(rootNode):

    rootNode.addObject('RequiredPlugin', name="plug1", pluginName='BeamAdapter Sofa.Component.Constraint.Projective Sofa.Component.LinearSolver.Direct Sofa.Component.ODESolver.Backward Sofa.Component.StateContainer Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Grid Sofa.Component.Visual Sofa.Component.SolidMechanics.Spring Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name="plug2", pluginName='Sofa.Component.AnimationLoop Sofa.Component.Collision.Detection.Algorithm Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Geometry Sofa.Component.Collision.Response.Contact Sofa.Component.Constraint.Lagrangian.Correction Sofa.Component.Constraint.Lagrangian.Solver Sofa.Component.IO.Mesh')
    rootNode.addObject('RequiredPlugin', pluginName='Sofa.Component.Topology.Mapping Sofa.Component.Mapping.Linear Sofa.GL.Component.Rendering3D')
    # rootNode.dt = 0.005   # 5 ms
    rootNode.addObject("VisualStyle", displayFlags="showVisualModels hideBehaviorModels showCollisionModels")
    rootNode.findData("bbox").value = "-0.10 -0.10 -0.05 0.10 0.10 0.50"

    rootNode.findData('dt').value = 0.005



    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.addObject('GenericConstraintSolver',
                       name='GCS',
                       maxIt=1000, tolerance=1e-6,
                       computeConstraintForces=True)


    MM = 1e-3  # millimeter in meters

    # --- Catheter material (SI) ---
    catheter_radius = 2.0 * MM          # 2 mm -> 0.002 m
    catheter_E      = 2000e6            # 2000 MPa -> 2e9 Pa
    catheter_rho    = 1.1e-6 * 1e9      # kg/mm^3 -> kg/m^3 => 1100 kg/m^3

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
        step  = 1.0 * MM,      # 1 mm
        speed = 2.0 * MM,      # 2 mm/s        
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
        rotation=(-20, 0, 75),
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