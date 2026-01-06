import Sofa
import os
import numpy as np

# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# MESH_DIR = os.path.join(THIS_DIR, "mesh")
# carotids_path = os.path.join(MESH_DIR, "carotids.stl")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(THIS_DIR, "mesh")

carotids_path        = os.path.join(MESH_DIR, "carotids.stl")
aneurysm_collis_path = os.path.join(MESH_DIR, "aneurysm3d_collis.obj")
aneurysm_surf_path   = os.path.join(MESH_DIR, "aneurysm3d_surface.obj")
cava_path            = os.path.join(MESH_DIR, "CavaVeinAndHeart.obj")
key_tip_path         = os.path.join(MESH_DIR, "key_tip.obj")
phantom_path         = os.path.join(MESH_DIR, "phantom.obj")

def _get_data_float(obj, name):
    """Robustly fetch a SOFA Data field as float, or None."""
    if obj is None:
        return None
    try:
        d = obj.findData(name)
        if d is not None:
            return float(d.value)
    except Exception:
        pass
    return None


class SceneUnitBanner(Sofa.Core.Controller):
    """
    Prints a single, explicit unit statement for this scene.
    You said lengths are in mm. With time in s, SOFA's force unit is kg·mm/s².
    That equals 1e-3 N, i.e. milli-Newton (mN).
    """
    def __init__(self, beam_forcefield=None, **kwargs):
        super().__init__(**kwargs)
        self.beam_ff = beam_forcefield
        self._printed = False

    def onAnimateBeginEvent(self, event):
        if self._printed:
            return
        self._printed = True

        ctx = self.getContext()
        dt = float(ctx.dt.value) if hasattr(ctx, "dt") else None

        rho = _get_data_float(self.beam_ff, "massDensity")
        # If rho ~ 1e-6 and lengths are mm, rho is kg/mm^3 (consistent with 1550 kg/m^3).
        rho_note = ""
        if rho is not None:
            if 1e-9 < rho < 1e-3:
                rho_note = " (looks like kg/mm^3; e.g., 1.55e-6 ≈ 1550 kg/m^3)"
            else:
                rho_note = " (not in the typical kg/mm^3 range; verify your density units)"

        print("[Units] ==================================================")
        print(f"[Units] Assumption: length = mm, time = s  (dt={dt})")
        print("[Units] Therefore: force unit in scene = kg·mm/s^2 = 1e-3 N = mN")
        print("[Units] Your logger prints forces in mN (and also N for convenience).")
        print(f"[Units] BeamForceField.massDensity = {rho}{rho_note}")
        print("[Units] ==================================================")

class TipRegionContactForceLogger(Sofa.Core.Controller):
    def __init__(self, collision_mo, constraint_solver,
                 contact_listener=None,
                 catheter_radius_mm=3.0,
                 k=5,
                 sample_every=1,
                 csv_path="force_log.csv",
                 make_plot=True,
                 plot_path="force_plot.png",
                 **kwargs):
        super().__init__(**kwargs)
        self.collision_mo = collision_mo
        self.solver = constraint_solver
        self.contact_listener = contact_listener
        self.catheter_radius = float(catheter_radius_mm)
        self.k = int(k)
        self.sample_every = int(sample_every)
        self.csv_path = str(csv_path)
        self.samples = []
        self.step = 0

    def _surface_point_from_theta(self, p_tip, e1, e2, theta_deg):
        if not np.isfinite(theta_deg):
            return None

        theta = np.deg2rad(theta_deg)
        r = self.catheter_radius

        return (
            p_tip
            + r * np.cos(theta) * e1
            + r * np.sin(theta) * e2
        )

    def _compute_tip_force_and_location(self, dt, force_eps_mN=1e-6):
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

        # per-point forces on last k points
        tip_ids = np.arange(tip_start, npts, dtype=int)
        Fi = np.zeros((tip_ids.size, 3), dtype=float)
        mags = np.zeros(tip_ids.size, dtype=float)

        for i, pid in enumerate(tip_ids):
            base = 3 * pid
            Fi[i, :] = [float(f[base+0]), float(f[base+1]), float(f[base+2])]
            mags[i] = np.linalg.norm(Fi[i, :])

        Fvec = Fi.sum(axis=0)              # mN
        Fx, Fy, Fz = map(float, Fvec)
        Fmag = float(np.linalg.norm(Fvec))

        # Active points for localisation
        active = mags > force_eps_mN
        active_pts = int(np.count_nonzero(active))

        # Force-weighted contact centroid on tip region (centerline points)
        Cp = None
        if active_pts > 0:
            P = pos[tip_ids[active]]
            w = mags[active]
            Cp = (P * w[:, None]).sum(axis=0) / w.sum()

        # Tip frame + circumferential angle from force direction (robust)
        theta_deg = np.nan
        Fperp_mN = 0.0
        frame = self._compute_tip_frame(pos)
        if frame is not None:
            p_tip, t_hat, e1, e2 = frame
            theta_deg, Fperp_mN = self._force_angle_about_tip(Fvec, t_hat, e1, e2)

        surface_pt = None
        if frame is not None and np.isfinite(theta_deg):
            p_tip, t_hat, e1, e2 = frame
            surface_pt = self._surface_point_from_theta(p_tip, e1, e2, theta_deg)


        return (
            Fx, Fy, Fz, Fmag,
            npts, tip_start,
            Cp, theta_deg, active_pts,
            Fperp_mN,
            surface_pt
        )
    def _v3_to_np(self, v):
        """
        Convert a SOFA Vec3d (or list/tuple/np array) into a numpy float array shape (3,).
        Returns None if conversion fails.
        """
        if v is None:
            return None

        # 1) Try indexable (v[0], v[1], v[2])
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except Exception:
            pass

        # 2) Try iterable (list(v) -> length 3)
        try:
            vv = list(v)
            if len(vv) >= 3:
                return np.array([float(vv[0]), float(vv[1]), float(vv[2])], dtype=float)
        except Exception:
            pass

        # 3) Try attribute access (v.x, v.y, v.z)
        try:
            return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
        except Exception:
            pass

        # 4) Last resort: numpy conversion
        try:
            a = np.asarray(v, dtype=float).ravel()
            if a.size >= 3:
                return a[:3].astype(float)
        except Exception:
            pass

        return None


    def _closest_contact_pair(self):
        """
        Returns (gap_mm, p1, p2, nContacts) where:
        - p1 is the point on collisionModel1 (catheter) in world coords (mm)
        - p2 is the point on collisionModel2 (vessel) in world coords (mm)
        If points cannot be parsed, p1/p2 are None but gap and nContacts are still returned.
        """
        cl = self.contact_listener
        if cl is None:
            return None

        # Number of contacts
        try:
            nC = int(cl.getNumberOfContacts())
        except Exception:
            return None
        if nC <= 0:
            return None

        # Pick closest (min gap) contact index
        try:
            dists = np.asarray(cl.getDistances(), dtype=float).ravel()
            if dists.size == 0:
                return None
            i = int(np.nanargmin(dists))
            gap = float(dists[i])
        except Exception:
            return None

        # Try to fetch contact points
        try:
            cps = cl.getContactPoints()
            if cps is None or len(cps) <= i:
                return (gap, None, None, nC)

            item = cps[i]

            # Expected common format: (id1, p1, id2, p2)
            if not (isinstance(item, (list, tuple)) and len(item) >= 4):
                return (gap, None, None, nC)

            p1 = self._v3_to_np(item[1])
            p2 = self._v3_to_np(item[3])

            # If conversion failed, keep them as None (but still return gap + count)
            if p1 is None or p2 is None:
                return (gap, None, None, nC)

            return (gap, p1, p2, nC)

        except Exception:
            return (gap, None, None, nC)



    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.sample_every > 1 and (self.step % self.sample_every != 0):
            return

        ctx = self.getContext()
        t = float(ctx.time.value)
        dt = float(ctx.dt.value)
        if dt <= 0:
            return

        out = self._compute_tip_force_and_location(dt)

        # stable CSV row template
        row = {
            "t_s": t,
            "Fx_mN": np.nan, "Fy_mN": np.nan, "Fz_mN": np.nan, "Fmag_mN": np.nan,
            "Fx_N":  np.nan, "Fy_N":  np.nan, "Fz_N":  np.nan, "Fmag_N":  np.nan,

            "tip_cpx_mm": np.nan, "tip_cpy_mm": np.nan, "tip_cpz_mm": np.nan,
            "tip_theta_deg": np.nan,
            "tip_activePts": 0,
            "Fperp_mN": np.nan,

            # NEW: inferred surface point on catheter radius (world coords)
            "tip_sx_mm": np.nan, "tip_sy_mm": np.nan, "tip_sz_mm": np.nan,

            "cp1x_mm": np.nan, "cp1y_mm": np.nan, "cp1z_mm": np.nan,
            "cp2x_mm": np.nan, "cp2y_mm": np.nan, "cp2z_mm": np.nan,
            "gap_mm": np.nan,
            "nContacts": 0,
        }

        if out is None:
            self.samples.append(row)
            if len(self.samples) % 100 == 0:
                self._write_csv()
            return

        # Support both return shapes:
        # old: (Fx, Fy, Fz, Fmag, npts, tip_start, Cp, theta_deg, active_pts, Fperp_mN)
        # new: (..., Fperp_mN, surface_pt)
        if len(out) == 10:
            Fx, Fy, Fz, Fmag, npts, tip_start, Cp, theta_deg, active_pts, Fperp_mN = out
            surface_pt = None
        elif len(out) >= 11:
            Fx, Fy, Fz, Fmag, npts, tip_start, Cp, theta_deg, active_pts, Fperp_mN, surface_pt = out[:11]
        else:
            # Unexpected return; log NaNs but don't crash
            self.samples.append(row)
            if len(self.samples) % 100 == 0:
                self._write_csv()
            return

        row.update({
            "Fx_mN": float(Fx), "Fy_mN": float(Fy), "Fz_mN": float(Fz), "Fmag_mN": float(Fmag),
            "Fx_N": float(Fx) * 1e-3, "Fy_N": float(Fy) * 1e-3, "Fz_N": float(Fz) * 1e-3, "Fmag_N": float(Fmag) * 1e-3,
            "tip_theta_deg": float(theta_deg) if np.isfinite(theta_deg) else np.nan,
            "tip_activePts": int(active_pts),
            "Fperp_mN": float(Fperp_mN) if np.isfinite(Fperp_mN) else np.nan,
        })

        if Cp is not None:
            row.update({
                "tip_cpx_mm": float(Cp[0]),
                "tip_cpy_mm": float(Cp[1]),
                "tip_cpz_mm": float(Cp[2]),
            })

        # Write inferred surface point (where on catheter radius)
        if surface_pt is not None:
            row.update({
                "tip_sx_mm": float(surface_pt[0]),
                "tip_sy_mm": float(surface_pt[1]),
                "tip_sz_mm": float(surface_pt[2]),
            })

        # Always set nContacts from the listener (ground truth)
        nC_total = self._n_contacts()
        row["nContacts"] = nC_total

        cc = self._closest_contact_pair()
        if cc is not None:
            gap, p1, p2, nC = cc
            row["gap_mm"] = float(gap)
            row["nContacts"] = int(nC_total)  # keep total from listener
            if p1 is not None and p2 is not None:
                row.update({
                    "cp1x_mm": float(p1[0]), "cp1y_mm": float(p1[1]), "cp1z_mm": float(p1[2]),
                    "cp2x_mm": float(p2[0]), "cp2y_mm": float(p2[1]), "cp2z_mm": float(p2[2]),
                })




        self.samples.append(row)
        if self.contact_listener is not None:
            nC = int(self.contact_listener.getNumberOfContacts())
            if nC > 0:
                cps = self.contact_listener.getContactPoints()
                print("[DBG] cps type:", type(cps), "len:", (len(cps) if cps is not None else None))
                if cps and len(cps) > 0:
                    print("[DBG] cps[0] =", cps[0])

        print(
            f"[TipRegion] t={t:.3f} tipPts={tip_start}..{npts-1} "
            f"|F|={Fmag:.3g} mN  Fperp={Fperp_mN:.3g} mN  "
            f"theta={row['tip_theta_deg']:.1f}deg  nContacts={row['nContacts']}  "
            f"tip_surface=({row['tip_sx_mm']:.1f},{row['tip_sy_mm']:.1f},{row['tip_sz_mm']:.1f})"
        )

        if len(self.samples) % 100 == 0:
            self._write_csv()

    def _n_contacts(self):
        cl = self.contact_listener
        if cl is None:
            return 0
        try:
            return int(cl.getNumberOfContacts())
        except Exception:
            return 0

    def _unit(self, v, eps=1e-12):
        n = float(np.linalg.norm(v))
        if n < eps:
            return None
        return v / n

    def _compute_tip_frame(self, pos):
        """
        Build a right-handed orthonormal frame at the tip.
        pos: (n,3) collision point positions in world frame.

        Returns: (p_tip, t_hat, e1, e2) or None if not enough info.
        """
        if pos.shape[0] < 2:
            return None

        p_tip = pos[-1]
        p_prev = pos[-2]
        t_hat = self._unit(p_tip - p_prev)
        if t_hat is None:
            return None

        # Choose a stable "up" vector not parallel to t_hat
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(up, t_hat)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        e1 = self._unit(np.cross(t_hat, up))
        if e1 is None:
            return None
        e2 = np.cross(t_hat, e1)  # already unit if t_hat and e1 are unit & orthogonal

        return p_tip, t_hat, e1, e2

    def _force_angle_about_tip(self, F, t_hat, e1, e2):
        """
        Compute circumferential angle of the lateral force component around the tip axis.
        F: 3-vector (mN)
        Returns theta_deg in [0,360) and lateral magnitude |F_perp| (mN)
        """
        # Remove axial component
        F_perp = F - np.dot(F, t_hat) * t_hat
        Fp = float(np.linalg.norm(F_perp))
        if Fp < 1e-12:
            return np.nan, 0.0

        x = float(np.dot(F_perp, e1))
        y = float(np.dot(F_perp, e2))
        theta = float(np.degrees(np.arctan2(y, x)))
        if theta < 0.0:
            theta += 360.0
        return theta, Fp

    def _write_csv(self):
        try:
            import csv
            if not self.samples:
                return

            fieldnames = list(self.samples[0].keys())
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in self.samples:
                    w.writerow(row)

            print(f"[TipRegion] CSV updated: {self.csv_path}  (rows={len(self.samples)})")
        except Exception as e:
            print(f"[TipRegion] CSV write failed: {e}")

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

    rootNode.addObject("VisualStyle", displayFlags="showVisualModels hideBehaviorModels showCollisionModels")
    rootNode.findData("bbox").value = "-200 -200 -200 200 200 600"


    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.addObject('GenericConstraintSolver',
                    name='GCS',
                    maxIt=1000, tolerance=1e-6,
                    computeConstraintForces=True)



    catheter_radius = 3      # mm
    catheter_E = 2000              # much lower than your 20000
    catheter_rho = 1.1e-6          # kg/mm^3 ~ 1100 kg/m^3 (polymer-ish)

    rootNode.addObject('CollisionPipeline', draw='0', depth='6', verbose='1')
    rootNode.addObject('BruteForceBroadPhase', name='N2')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('LocalMinDistance', contactDistance=str(catheter_radius), alarmDistance=str(2*catheter_radius), name='localmindistance', angleCone='0.2')
    rootNode.addObject('CollisionResponse', name='Response', response='FrictionContactConstraint')


    topoLines = rootNode.addChild('EdgeTopology')


    topoLines.addObject('RodStraightSection', name='StraightSection',
                        length=980.0, radius=catheter_radius,
                        nbBeams=50, nbEdgesCollis=50, nbEdgesVisu=200,
                        youngModulus=catheter_E, massDensity=catheter_rho, poissonRatio=0.45)



    # topoLines.addObject('RodSpireSection', name='SpireSection', 
    #                              length=10.0, radius=catheter_radius, 
    #                              nbBeams=10, nbEdgesCollis=10, nbEdgesVisu=200,
    #                              spireDiameter=100, spireHeight=0,
    #                              youngModulus=20000, massDensity=0.00000155, poissonRatio=0.3)
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
                                    p0=[0,0,0])
    BeamMechanics.addObject('MechanicalObject', showIndices=False, name='DOFs', template='Rigid3d', ry=-90)
    BeamMechanics.addObject('WireBeamInterpolation', name='BeamInterpolation', WireRestShape='@../EdgeTopology/BeamRestShape', printLog=False)
    BeamMechanics.addObject('AdaptiveBeamForceFieldAndMass', name='BeamForceField', massDensity=catheter_rho, interpolation='@BeamInterpolation')
    BeamMechanics.addObject(
        'InterventionalRadiologyController',
        name='DeployController',
        template='Rigid3d',
        instruments='BeamInterpolation',
        topology='@MeshLines',
        startingPos=[0,0,0,0,0,0,1],
        xtip=[0],
        printLog=True,
        rotationInstrument=0,  
        step=5., speed=10.,
        listening=True,
        controlledInstrument=0
    )

    BeamMechanics.addObject('LinearSolverConstraintCorrection', wire_optimization='true', printLog=False)
    BeamMechanics.addObject('FixedProjectiveConstraint', indices=0, name='FixedConstraint')
    BeamMechanics.addObject('RestShapeSpringsForceField', points='@DeployController.indexFirstNode', angularStiffness=1e8, stiffness=1e8)


    BeamCollis = BeamMechanics.addChild('CollisionModel')
    BeamCollis.activated = True
    BeamCollis.addObject('EdgeSetTopologyContainer', name='collisEdgeSet')
    BeamCollis.addObject('EdgeSetTopologyModifier', name='colliseEdgeModifier')
    BeamCollis.addObject('MechanicalObject', name='CollisionDOFs')
    BeamCollis.addObject('MultiAdaptiveBeamMapping', controller='../DeployController', useCurvAbs=True, printLog=False, name='collisMap')
    # BeamCollis.addObject('LineCollisionModel', contactDistance=catheter_radius)
    # BeamCollis.addObject('PointCollisionModel', contactDistance=catheter_radius)
    BeamCollis.addObject('LineCollisionModel', name='cathLine', proximity=0.0)
    BeamCollis.addObject('PointCollisionModel', name='cathPoints', proximity=0.0)



    gcs = rootNode.getObject('GCS')
    collision_mo = BeamCollis.getObject('CollisionDOFs')
    beam_ff = BeamMechanics.getObject('BeamForceField')

    rootNode.addObject(SceneUnitBanner(beam_forcefield=beam_ff))

    # Put files next to this scene
    csv_out = os.path.join(THIS_DIR, "force_log.csv")
    png_out = os.path.join(THIS_DIR, "force_plot.png")

    # BeamCollis.addObject(TipRegionContactForceLogger(
    #     collision_mo=collision_mo,
    #     constraint_solver=gcs,
    #     contact_listener=contact_listener,
    #     k=5,
    #     sample_every=1,
    #     csv_path=csv_out,
    #     make_plot=True,
    #     plot_path=png_out
    # ))

    # print("[DBG] cathLine:", BeamCollis.getObject('cathLine'))
    # print("[DBG] cathPoints:", BeamCollis.getObject('cathPoints'))
    # print("[DBG] vesselTris:", Carotids.getObject('vesselTris'))

    # print("[DBG] cathLine path:", BeamCollis.getObject('cathLine').getPathName())
    # print("[DBG] vesselTris path:", Carotids.getObject('vesselTris').getPathName())
    # --- Catheter tube visualisation (surface) ---
    VisuCath = BeamMechanics.addChild('VisuCatheter')

    # A mechanical object to carry the visual vertices
    VisuCath.addObject('MechanicalObject', name="Quads", template="Vec3d")

    # Quad topology that will represent the tube
    VisuCath.addObject('QuadSetTopologyContainer', name="TubeQuads")
    VisuCath.addObject('QuadSetTopologyModifier', name="TubeQuadsModifier")
    VisuCath.addObject('QuadSetGeometryAlgorithms', name="TubeGeom", template="Vec3d")

    # Build a tube (quads) around the beam centerline edges
    # IMPORTANT: "input" must reference an Edge topology.
    # In many BeamAdapter scenes, the EdgeSetTopologyContainer under your EdgeTopology node is used.
    VisuCath.addObject(
        'Edge2QuadTopologicalMapping',
        name="Edge2Quad",
        nbPointsOnEachCircle=12,           # smoother tube
        radius=catheter_radius,            # your catheter radius in mm
        input='@../../EdgeTopology/meshLines',
        output='@TubeQuads',
        flipNormals=True,
        printLog=False
    )

    # Map the tube vertices motion from the beam interpolation
    VisuCath.addObject(
        'AdaptiveBeamMapping',
        name="VisuMapCath",
        useCurvAbs=True,
        isMechanical=False,
        interpolation='@../BeamInterpolation',
        printLog=False
    )

    # Render it
    VisuOgl = VisuCath.addChild('Ogl')
    VisuOgl.addObject('OglModel', name="CatheterVisual", src='@../TubeQuads', color='white')
    VisuOgl.addObject('IdentityMapping', input='@../Quads', output='@CatheterVisual')

    # Carotids = rootNode.addChild('Carotids')
    # Carotids.addObject('MeshSTLLoader', filename=carotids_path, flipNormals=False, triangulate=True,
    #                 name='meshLoader', rotation=[10.0, 0.0, -90.0])
    # Carotids.addObject('MeshTopology', position='@meshLoader.position', triangles='@meshLoader.triangles')
    # Carotids.addObject('MechanicalObject', position=[0,0,400], scale=3, name='DOFs1', ry=90)
    # Carotids.addObject('LineCollisionModel', name='vesselLines', moving=False, simulated=False)
    # Carotids.addObject('TriangleCollisionModel', name='vesselTris', moving=False, simulated=False)
    # print("[DBG] vesselTris:", Carotids.getObject('vesselTris'), flush=True)
    # print("[DBG] vesselTris path:", Carotids.getObject('vesselTris').getPathName(), flush=True)

        # --- create ContactListener NOW ---
    # cath_points = BeamCollis.getObject('cathPoints')


    # # Carotids (you already have this; shown here in the unified style)
    # carotids = add_static_mesh(
    #     rootNode, "Carotids",
    #     carotids_path,
    #     translation=(0,0,0),
    #     rotation=(0, 0, 0),
    #     scale=3.0,
    #     visual=True,
    #     collision=True,
    #     triangulate=True
    # )
    # # Example: listen against Carotids triangle collision model
    # vessel_tris = carotids.getObject('triColl')
# --- Carotids ---
    aneurysm_coll = add_static_mesh(
        rootNode, "AneurysmCollis",
        aneurysm_collis_path,
        translation=(0,0,400),
        rotation=(10, 0, -90),
        scale=3.0,
        visual=False,
        collision=True,
        triangulate=True
    )
    print("[DBG] catheter DOFs first:", BeamMechanics.getObject('DOFs').position[0], flush=True)
    print("[DBG] vessel triColl path:", vessel_tris.getPathName(), flush=True)
    print("[DBG] carotids dofs:", aneurysm_coll.getObject('dofs').translation.value, aneurysm_coll.getObject('dofs').rotation.value, flush=True)

    # --- ContactListener (ONLY ONCE) ---
    cath_points = BeamCollis.getObject('cathPoints')
    vessel_tris = aneurysm_coll.getObject('triColl')

    cm1 = '@' + cath_points.getPathName()
    cm2 = '@' + vessel_tris.getPathName()

    contact_listener = rootNode.addObject(
        'ContactListener',
        name='CathVesselContactListener',
        collisionModel1=cm1,
        collisionModel2=cm2,
        listening=True
    )



    BeamCollis.addObject(TipRegionContactForceLogger(
    collision_mo=collision_mo,
    constraint_solver=gcs,
    contact_listener=contact_listener,
    k=5,
    sample_every=1,
    csv_path=csv_out,
    make_plot=True,
    plot_path=png_out
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
