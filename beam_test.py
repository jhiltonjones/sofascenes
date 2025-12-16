import Sofa
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(THIS_DIR, "mesh")
carotids_path = os.path.join(MESH_DIR, "carotids.stl")


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
    """
    Computes contact force on the last k collision points:
      f ≈ (J^T * constraintForces) / dt

    With mm–kg–s units, the resulting force is in mN (because kg·mm/s² = mN).
    """
    def __init__(self, collision_mo, constraint_solver, k=5, print_every=50,
                 debug_shapes=False, debug_lambda_stats=False, **kwargs):
        super().__init__(**kwargs)
        self.collision_mo = collision_mo
        self.solver = constraint_solver
        self.k = int(k)
        self.print_every = int(print_every)
        self.debug_shapes = bool(debug_shapes)
        self.debug_lambda_stats = bool(debug_lambda_stats)
        self.step = 0

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.step % self.print_every != 0:
            return

        ctx = self.getContext()
        t = float(ctx.time.value)
        dt = float(ctx.dt.value)
        if dt <= 0:
            return

        # Collision points
        pos = self.collision_mo.position.value
        npts = len(pos)
        if npts == 0:
            print(f"[TipRegion] t={t:.3f} npts=0 (no collision points)")
            return

        tip_start = max(0, npts - self.k)

        # Constraint Jacobian-like matrix (rows = constraints, cols = 3*npts)
        J = self.collision_mo.constraint.value
        if J is None or getattr(J, "shape", (0, 0))[0] == 0:
            print(f"[TipRegion] t={t:.3f} J is empty/unavailable")
            return

        # constraintForces from solver (often impulse-like, hence division by dt)
        lambdas = np.asarray(self.solver.constraintForces.value, dtype=float).ravel()
        if lambdas.size == 0:
            print(f"[TipRegion] t={t:.3f} constraintForces empty")
            return

        m = min(J.shape[0], lambdas.size)
        if m == 0:
            print(f"[TipRegion] t={t:.3f} m=0 (no active constraints)")
            return

        # Map to nodal force vector (size should be 3*npts)
        f = (J[:m, :].T @ lambdas[:m]) / dt

        # Verify expected sizing (for Vec3 collision DOFs)
        expected = 3 * npts
        if f.size < expected:
            print(f"[TipRegion] t={t:.3f} f.size={f.size} < expected={expected} (unexpected DOF layout)")
            return

        Fx = Fy = Fz = 0.0
        for pid in range(tip_start, npts):
            base = 3 * pid
            Fx += float(f[base + 0])
            Fy += float(f[base + 1])
            Fz += float(f[base + 2])

        Fmag = float(np.sqrt(Fx*Fx + Fy*Fy + Fz*Fz))

        # Units:
        # - scene force unit = kg·mm/s²  ==  mN
        # - Newtons = mN * 1e-3
        Fx_mN, Fy_mN, Fz_mN, Fmag_mN = Fx, Fy, Fz, Fmag
        Fx_N,  Fy_N,  Fz_N,  Fmag_N  = Fx_mN * 1e-3, Fy_mN * 1e-3, Fz_mN * 1e-3, Fmag_mN * 1e-3

        print(f"[TipRegion] t={t:.3f} tipPts={tip_start}..{npts-1} "
              f"F={Fx_mN:.6g},{Fy_mN:.6g},{Fz_mN:.6g} mN |F|={Fmag_mN:.6g} mN "
              f"(= {Fmag_N:.6g} N)")

        if self.debug_shapes:
            print(f"[TipRegion][DBG] dt={dt:g} npts={npts} J.shape={getattr(J,'shape',None)} "
                  f"lambdas.size={lambdas.size} m={m} f.size={f.size}")

        if self.debug_lambda_stats:
            lm = lambdas[:m]
            print(f"[TipRegion][DBG] constraintForces stats: min={lm.min():.6g} max={lm.max():.6g} "
                  f"norm={np.linalg.norm(lm):.6g} (constraint space, impulse-like in many pipelines)")




def createScene(rootNode):

    rootNode.addObject('RequiredPlugin', name="plug1", pluginName='BeamAdapter Sofa.Component.Constraint.Projective Sofa.Component.LinearSolver.Direct Sofa.Component.ODESolver.Backward Sofa.Component.StateContainer Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Grid Sofa.Component.Visual Sofa.Component.SolidMechanics.Spring Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name="plug2", pluginName='Sofa.Component.AnimationLoop Sofa.Component.Collision.Detection.Algorithm Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Geometry Sofa.Component.Collision.Response.Contact Sofa.Component.Constraint.Lagrangian.Correction Sofa.Component.Constraint.Lagrangian.Solver Sofa.Component.IO.Mesh')
    rootNode.addObject('VisualStyle',
                    displayFlags='showVisualModels hideBehaviorModels showCollisionModels hideMappings showInteractionForceFields')

    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.addObject('GenericConstraintSolver',
                    name='GCS',
                    maxIt=1000, tolerance=1e-6,
                    computeConstraintForces=True)





    rootNode.addObject('CollisionPipeline', draw='0', depth='6', verbose='1')
    rootNode.addObject('BruteForceBroadPhase', name='N2')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('LocalMinDistance', contactDistance='0.1', alarmDistance='2', name='localmindistance', angleCone='0.2')
    rootNode.addObject('CollisionResponse', name='Response', response='FrictionContactConstraint')


    topoLines = rootNode.addChild('EdgeTopology')
    catheter_radius = 2       # mm
    catheter_E = 2000              # much lower than your 20000
    catheter_rho = 1.1e-6          # kg/mm^3 ~ 1100 kg/m^3 (polymer-ish)

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
    BeamMechanics.addObject('AdaptiveBeamForceFieldAndMass', name='BeamForceField', massDensity=0.00000155, interpolation='@BeamInterpolation')
    BeamMechanics.addObject('InterventionalRadiologyController', name='DeployController', template='Rigid3d', instruments='BeamInterpolation', 
                                    topology="@MeshLines", startingPos=[0, 0, 0, 0, 0, 0, 1], xtip=[0], printLog=True, 
                                    rotationInstrument=[0,0], step=1., speed=5., 
                                    listening=True, controlledInstrument=0)
    BeamMechanics.addObject('LinearSolverConstraintCorrection', wire_optimization='true', printLog=False)
    BeamMechanics.addObject('FixedProjectiveConstraint', indices=0, name='FixedConstraint')
    BeamMechanics.addObject('RestShapeSpringsForceField', points='@DeployController.indexFirstNode', angularStiffness=1e8, stiffness=1e8)


    BeamCollis = BeamMechanics.addChild('CollisionModel')
    BeamCollis.activated = True
    BeamCollis.addObject('EdgeSetTopologyContainer', name='collisEdgeSet')
    BeamCollis.addObject('EdgeSetTopologyModifier', name='colliseEdgeModifier')
    BeamCollis.addObject('MechanicalObject', name='CollisionDOFs')
    BeamCollis.addObject('MultiAdaptiveBeamMapping', controller='../DeployController', useCurvAbs=True, printLog=False, name='collisMap')
    BeamCollis.addObject('LineCollisionModel', proximity=0.0)
    BeamCollis.addObject('PointCollisionModel', proximity=0.0)
    gcs = rootNode.getObject('GCS')
    collision_mo = BeamCollis.getObject('CollisionDOFs')
    beam_ff = BeamMechanics.getObject('BeamForceField')

    rootNode.addObject(SceneUnitBanner(beam_forcefield=beam_ff))

    BeamCollis.addObject(TipRegionContactForceLogger(
        collision_mo=collision_mo,
        constraint_solver=gcs,
        k=5,
        print_every=50,
        debug_shapes=True,        
        debug_lambda_stats=False   
    ))

    Carotids = rootNode.addChild('Carotids')
    Carotids.addObject('MeshSTLLoader', filename=carotids_path, flipNormals=False, triangulate=True, name='meshLoader', rotation=[10.0, 0.0, -90.0])
    Carotids.addObject('MeshTopology', position='@meshLoader.position', triangles='@meshLoader.triangles')
    Carotids.addObject('MechanicalObject', position=[0,0,400], scale=3, name='DOFs1', ry=90)
    Carotids.addObject('TriangleCollisionModel', moving=False, simulated=False)
    Carotids.addObject('LineCollisionModel', moving=False, simulated=False)

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
