import Sofa
import os
import numpy as np


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
        self.debug = True
        self.debug_every = 10
        self.eps_perp = 1e-6   # mN
        self.eps_axis = 1e-9   # mm

    def _wrap_deg(self, a):
        a = float(a)
        a = a % 360.0
        if a < 0:
            a += 360.0
        return a

    def _angdiff_deg(self, a, b):
        """Smallest signed difference a-b in degrees, range (-180,180]."""
        d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
        return d

    def _debug_frame_invariants(self, tag, p_tip, t_hat, e1, e2):
        # norms
        nt = np.linalg.norm(t_hat)
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        # orthogonality
        dt1 = float(np.dot(t_hat, e1))
        dt2 = float(np.dot(t_hat, e2))
        d12 = float(np.dot(e1, e2))
        # right-handedness check: cross(t_hat,e1) should align with e2
        rh = float(np.dot(np.cross(t_hat, e1), e2))
        print(f"[DBG:{tag}] frame norms: |t|={nt:.6g} |e1|={n1:.6g} |e2|={n2:.6g}")
        print(f"[DBG:{tag}] orth: t·e1={dt1:.3g} t·e2={dt2:.3g} e1·e2={d12:.3g}  RH(cross(t,e1)·e2)={rh:.6g}")
        print(f"[DBG:{tag}] p_tip={p_tip}")

    def _debug_force_decomp(self, tag, Fvec, t_hat):
        Fax = float(np.dot(Fvec, t_hat))
        Fperp_vec = Fvec - Fax * t_hat
        Fperp = float(np.linalg.norm(Fperp_vec))
        Fmag = float(np.linalg.norm(Fvec))
        recon = np.sqrt(Fax*Fax + Fperp*Fperp)
        print(f"[DBG:{tag}] |F|={Fmag:.6g}  Fax={Fax:.6g}  Fperp={Fperp:.6g}  sqrt(Fax^2+Fperp^2)={recon:.6g}  err={abs(Fmag-recon):.3g}")
        return Fax, Fperp_vec, Fperp

    def _debug_surface_point(self, tag, p_tip, surface_pt):
        if surface_pt is None:
            print(f"[DBG:{tag}] surface_pt=None")
            return
        r = float(np.linalg.norm(surface_pt - p_tip))
        print(f"[DBG:{tag}] |surface - p_tip| = {r:.6g} mm (expected {self.catheter_radius:.6g})  err={abs(r-self.catheter_radius):.3g}")

    def _debug_theta_surface_consistency(self, tag, theta_deg, e1, e2, u_dir):
        # theta should match atan2(u·e2, u·e1)
        x = float(np.dot(u_dir, e1))
        y = float(np.dot(u_dir, e2))
        theta2 = self._wrap_deg(np.degrees(np.arctan2(y, x)))
        dth = self._angdiff_deg(theta_deg, theta2)
        print(f"[DBG:{tag}] theta={theta_deg:.3f}  recomputed_theta={theta2:.3f}  dtheta={dth:.3g} deg")
        print(f"[DBG:{tag}] u_dir={u_dir}  (u·e1={x:.6g}, u·e2={y:.6g})")

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
        # Tip frame + circumferential angle from force direction
        theta_deg = np.nan
        Fperp_mN = 0.0
        frame = self._compute_tip_frame(pos)

        if frame is not None:
            p_tip, t_hat, e1, e2 = frame
            theta_deg, Fperp_mN = self._force_angle_about_tip(Fvec, t_hat, e1, e2)

        surface_pt = None
        if frame is not None and np.isfinite(theta_deg) and Fperp_mN > self.eps_perp:
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
        Returns (gap_mm, p1, p2, nContacts, id1, id2) where:
        - id1 is index on collisionModel1 (catheter points)
        - id2 is index on collisionModel2 (vessel triangles, etc.)
        - p1 is point on collisionModel1 in world coords (often proxy centerline for PointCollisionModel)
        - p2 is point on collisionModel2 in world coords (on vessel surface)
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
                return (gap, None, None, nC, None, None)

            item = cps[i]
            if not (isinstance(item, (list, tuple)) and len(item) >= 4):
                return (gap, None, None, nC, None, None)

            # Typical format: (id1, p1, id2, p2)
            try:
                id1 = int(item[0])
            except Exception:
                id1 = None
            try:
                id2 = int(item[2])
            except Exception:
                id2 = None

            p1 = self._v3_to_np(item[1])
            p2 = self._v3_to_np(item[3])

            return (gap, p1, p2, nC, id1, id2)

        except Exception:
            return (gap, None, None, nC, None, None)



    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.sample_every > 1 and (self.step % self.sample_every != 0):
            return

        ctx = self.getContext()
        t = float(ctx.time.value)
        dt = float(ctx.dt.value)
        if dt <= 0:
            return

        # --- Base row (stable schema; always write same columns) ---
        row = {
            "t_s": t,

            # total force at tip region (mN + N)
            "Fx_mN": np.nan, "Fy_mN": np.nan, "Fz_mN": np.nan, "Fmag_mN": np.nan,
            "Fx_N":  np.nan, "Fy_N":  np.nan, "Fz_N":  np.nan, "Fmag_N":  np.nan,

            # tip-region localisation (force-weighted centroid on centerline proxy)
            "tip_cpx_mm": np.nan, "tip_cpy_mm": np.nan, "tip_cpz_mm": np.nan,
            "tip_activePts": 0,

            # force-based circumferential info (about tip tangent)
            "tip_theta_deg": np.nan,
            "Fperp_mN": np.nan,

            # inferred surface point from force-based theta (ideal cylinder)
            "tip_sx_mm": np.nan, "tip_sy_mm": np.nan, "tip_sz_mm": np.nan,

            # closest contact pair from listener
            "gap_mm": np.nan,
            "nContacts": 0,
            "cp1x_mm": np.nan, "cp1y_mm": np.nan, "cp1z_mm": np.nan,
            "cp2x_mm": np.nan, "cp2y_mm": np.nan, "cp2z_mm": np.nan,

            # geometry-based cross-check from p2
            "cp_theta_deg": np.nan,
            "cp_sx_mm": np.nan, "cp_sy_mm": np.nan, "cp_sz_mm": np.nan,
            "dperp_norm_mm": np.nan,
            "theta_diff_deg": np.nan,
        }

        # --- Contact count is always available ---
        row["nContacts"] = int(self._n_contacts())

        # --- Closest contact pair (gap, cp1/cp2) if available ---
        cc = self._closest_contact_pair()
        if cc is not None:
            gap, p1, p2, _nC, id1, id2 = cc
            row["gap_mm"] = float(gap)
            if p1 is not None:
                row["cp1x_mm"], row["cp1y_mm"], row["cp1z_mm"] = map(float, p1)
            if p2 is not None:
                row["cp2x_mm"], row["cp2y_mm"], row["cp2z_mm"] = map(float, p2)

        # --- Compute tip-region constraint force resultants ---
        out = self._compute_tip_force_and_location(dt)
        if out is None:
            self.samples.append(row)
            if len(self.samples) % 100 == 0:
                self._write_csv()
            return

        # Unpack (support both old and new return shapes)
        surface_pt = None
        if len(out) == 10:
            Fx, Fy, Fz, Fmag, npts, tip_start, Cp, theta_deg, active_pts, Fperp_mN = out
        elif len(out) >= 11:
            Fx, Fy, Fz, Fmag, npts, tip_start, Cp, theta_deg, active_pts, Fperp_mN, surface_pt = out[:11]
        else:
            # Unexpected return; log row as-is
            self.samples.append(row)
            if len(self.samples) % 100 == 0:
                self._write_csv()
            return

        # Fill force totals
        row["Fx_mN"], row["Fy_mN"], row["Fz_mN"], row["Fmag_mN"] = map(float, [Fx, Fy, Fz, Fmag])
        row["Fx_N"],  row["Fy_N"],  row["Fz_N"],  row["Fmag_N"]  = [float(Fx) * 1e-3, float(Fy) * 1e-3, float(Fz) * 1e-3, float(Fmag) * 1e-3]

        # Fill tip-region values
        row["tip_activePts"] = int(active_pts)
        row["tip_theta_deg"] = float(theta_deg) if np.isfinite(theta_deg) else np.nan
        row["Fperp_mN"] = float(Fperp_mN) if np.isfinite(Fperp_mN) else np.nan

        if Cp is not None:
            row["tip_cpx_mm"], row["tip_cpy_mm"], row["tip_cpz_mm"] = map(float, Cp)

        if surface_pt is not None:
            row["tip_sx_mm"], row["tip_sy_mm"], row["tip_sz_mm"] = map(float, surface_pt)

        # --- Geometry-based cross-check (p2 -> cp_theta, cp_surface) AFTER theta_deg is known ---
        if cc is not None:
            gap, p1, p2, _nC, id1, id2 = cc
            if p2 is not None:
                pos = np.asarray(self.collision_mo.position.value, dtype=float)
                frame_tip = self._compute_tip_frame(pos)
                if frame_tip is not None:
                    p_tip, t_hat, e1, e2 = frame_tip

                    d = p2 - p_tip
                    d_perp = d - np.dot(d, t_hat) * t_hat
                    dperp_norm = float(np.linalg.norm(d_perp))
                    row["dperp_norm_mm"] = dperp_norm

                    cp_theta, cp_surface = self._theta_and_surface_from_cp2(
                        c=p_tip, t_hat=t_hat, e1=e1, e2=e2, p2=p2, eps=self.eps_axis
                    )

                    if np.isfinite(cp_theta):
                        row["cp_theta_deg"] = float(cp_theta)
                        if np.isfinite(theta_deg):
                            row["theta_diff_deg"] = float(self._angdiff_deg(theta_deg, cp_theta))

                    if cp_surface is not None:
                        row["cp_sx_mm"], row["cp_sy_mm"], row["cp_sz_mm"] = map(float, cp_surface)

                    if getattr(self, "debug", False) and (self.step % getattr(self, "debug_every", 10) == 0):
                        print(
                            f"[DBG] dperp_norm={row['dperp_norm_mm']:.6g}mm "
                            f"theta_force={row['tip_theta_deg']:.3f} cp_theta={row['cp_theta_deg']:.3f} "
                            f"diff={row['theta_diff_deg']:.3g}"
                        )

        # --- Append and write periodically ---
        self.samples.append(row)

        print(
            f"[TipRegion] t={t:.3f} tipPts={tip_start}..{npts-1} "
            f"|F|={row['Fmag_mN']:.3g} mN  Fperp={row['Fperp_mN']:.3g} mN  "
            f"theta={row['tip_theta_deg']:.1f}deg  nContacts={row['nContacts']}  "
            f"tip_surface=({row['tip_sx_mm']:.1f},{row['tip_sy_mm']:.1f},{row['tip_sz_mm']:.1f}) "
            f"gap={row['gap_mm']:.3g}mm"
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
    def _compute_frame_at_index(self, pos, idx):
        """
        Build a right-handed orthonormal frame at centerline point pos[idx].
        Returns (c, t_hat, e1, e2) or None.
        """
        n = pos.shape[0]
        if n < 2:
            return None
        if idx is None:
            return None

        idx = int(np.clip(idx, 0, n - 1))
        c = pos[idx]

        # Tangent from neighbors
        if idx == 0:
            a, b = pos[0], pos[1]
        elif idx == n - 1:
            a, b = pos[n - 2], pos[n - 1]
        else:
            a, b = pos[idx - 1], pos[idx + 1]

        t_hat = self._unit(b - a)
        if t_hat is None:
            return None

        # Stable cross-section basis
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(up, t_hat)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        e1 = self._unit(np.cross(t_hat, up))
        if e1 is None:
            return None
        e2 = np.cross(t_hat, e1)

        return c, t_hat, e1, e2


    def _theta_and_surface_from_cp2(self, c, t_hat, e1, e2, p2, eps=1e-9):
        """
        Use vessel point p2 to infer circumferential angle and surface contact point on ideal cylinder.
        Returns (theta_deg, surface_pt) or (nan, None) if ill-conditioned.
        """
        if c is None or t_hat is None or e1 is None or e2 is None or p2 is None:
            return np.nan, None

        d = p2 - c
        d_perp = d - np.dot(d, t_hat) * t_hat
        n = float(np.linalg.norm(d_perp))
        if n < eps:
            # Vessel point is almost on axis direction: 'side' is not well-defined
            return np.nan, None

        r_hat = d_perp / n

        # Angle in cross-section basis
        x = float(np.dot(r_hat, e1))
        y = float(np.dot(r_hat, e2))
        theta = float(np.degrees(np.arctan2(y, x)))
        if theta < 0.0:
            theta += 360.0

        # Surface point on ideal cylinder
        surface_pt = c + self.catheter_radius * r_hat
        return theta, surface_pt

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



    catheter_radius = 2   
    catheter_E = 2000      
    catheter_rho = 1.1e-6          

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

    BeamCollis.addObject('LineCollisionModel', name='cathLine', proximity=0.0)
    BeamCollis.addObject('PointCollisionModel', name='cathPoints', proximity=0.0)



    gcs = rootNode.getObject('GCS')
    collision_mo = BeamCollis.getObject('CollisionDOFs')
    beam_ff = BeamMechanics.getObject('BeamForceField')

    rootNode.addObject(SceneUnitBanner(beam_forcefield=beam_ff))

    
    csv_out = os.path.join(THIS_DIR, "force_log.csv")
    png_out = os.path.join(THIS_DIR, "force_plot.png")

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

    carotids = add_static_mesh(
        rootNode, "Carotids",
        carotids_path,
        translation=(0, 0, 0),
        rotation=(-30, 90, -90),
        scale=3.0,
        visual=True,
        collision=True,     
        triangulate=True
    )

    cath_points = BeamCollis.getObject('cathPoints')
    vessel_tris = carotids.getObject('triColl')

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
    k=1,
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
