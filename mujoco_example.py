import mujoco
import mujoco.viewer
import time
import numpy as np

# ==========================================
# Parameter Configuration Section (Modify here to experiment)
# ==========================================
class Config:
    # --- 1. Geometry and Mass ---
    OBJ_RADIUS = 0.05      # Cylinder radius (m)
    OBJ_HEIGHT = 0.2       # Cylinder height (m)
    OBJ_DENSITY = 2000     # Object density (kg/m^3)
    
    # --- 2. Contact Friction ---
    # friction = [sliding, torsional, rolling]
    # Theoretically: Sliding friction coeff mu >= (m*g) / (2 * gripping force) to hold it
    FRICTION = "1.0 0.1 0.0001" 
    CONDIM = 4
    
    # --- 3. Contact Stiffness (Solref/Solimp) ---
    # 0.02 = Soft finger (20ms time constant), 0.002 = Hard finger
    SOLREF = "0.002 1"     
    
    SOLIMP = ".9 .9999 .001 .5 2"
    
    # Contact Margin: 1mm buffer
    MARGIN = "0.001" 

    # --- 4. Applied Force ---
    # Force applied on each side in Newtons (N)
    GRIP_FORCE = 20.0


# ==========================================
# 🛠️ Automatically Generate MJCF XML 
# ==========================================
def get_xml():
    xml = f"""
    <mujoco model="friction_test">
        <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast" impratio="100000" noslip_iterations="10"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
            <map force="0.05" znear="0.01"/>
        </visual>

        <default>
            <geom condim="{Config.CONDIM}" 
                  friction="{Config.FRICTION}" 
                  solref="{Config.SOLREF}" 
                  solimp="{Config.SOLIMP}"
                  margin="{Config.MARGIN}"
                  rgba="0.8 0.6 0.4 1"/>
        </default>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.2 0.2 1"/>

            <body name="object" pos="0 0 0.5">
                <freejoint/>
                <geom name="obj_geom" type="box" size="{Config.OBJ_RADIUS} {Config.OBJ_RADIUS} {Config.OBJ_HEIGHT/2}" 
                      density="{Config.OBJ_DENSITY}" rgba="0.3 0.9 0.3 1"/>
            </body>

            <body name="left_finger" pos="-0.15 0 0.5">
                <joint name="slide_left" type="slide" axis="1 0 0" damping="0"/>
                <geom name="left_geom" type="sphere" size="0.02" rgba="0.9 0.3 0.3 1"/>
                <site name="left_sensor" pos="0 0 0" size="0.01" rgba="1 0 0 0.5" group="2"/>
            </body>

            <body name="right_finger" pos="0.15 0 0.5">
                <joint name="slide_right" type="slide" axis="-1 0 0" damping="0"/>
                <geom name="right_geom" type="sphere" size="0.02" rgba="0.9 0.3 0.3 1"/>
                <site name="right_sensor" pos="0 0 0" size="0.01" rgba="1 0 0 0.5" group="2"/>
            </body>
        </worldbody>

        <actuator>
            <motor name="push_left" joint="slide_left" gear="1"/>
            <motor name="push_right" joint="slide_right" gear="1"/>
        </actuator>
        
        <sensor>
            <force name="force_left" site="left_sensor"/>
            <force name="force_right" site="right_sensor"/>
        </sensor>
    </mujoco>
    """
    return xml


# ==========================================
# 🚀 Main Run Loop
# ==========================================
def main():
    print(">>> Building MuJoCo model...")
    xml_content = get_xml()
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    # --- Physics Calculation ---
    volume = (Config.OBJ_RADIUS**2) * Config.OBJ_HEIGHT * 4
    mass = volume * Config.OBJ_DENSITY
    gravity_force = mass * 9.81
    
    # Friction threshold calculation
    mu_sliding = float(Config.FRICTION.split()[0])
    max_static_friction = 2 * mu_sliding * Config.GRIP_FORCE

    print("\n" + "="*50)
    print(f"Physics Parameters Overview:")
    print(f"  - Object Mass: {mass:.3f} kg")
    print(f"  - Gravity (G): {gravity_force:.3f} N")
    print(f"  - Single-side Grip Force: {Config.GRIP_FORCE:.1f} N")
    print(f"  - Friction Coefficient: {mu_sliding}")
    print(f"  - Max Static Friction (2*mu*N): {max_static_friction:.3f} N")
    print("-" * 50)
    
    if max_static_friction > gravity_force:
        safety_factor = max_static_friction / gravity_force
        print(f"Theoretical Result: Firmly held (Safety Factor {safety_factor:.2f})")
    else:
        print(f"Theoretical Result: Inevitable slip (Lacking {gravity_force - max_static_friction:.2f} N)")
    print("="*50 + "\n")

    # Launch Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable contact force visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        model.vis.scale.forcewidth = 0.05
        model.vis.map.force = 0.05 

        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()

            # 1. Apply grip force (continuous application)
            data.ctrl[0] = Config.GRIP_FORCE 
            data.ctrl[1] = Config.GRIP_FORCE

            # 2. Physics step
            mujoco.mj_step(model, data)

            # 3. Monitor status (print every timestep)
            if True:
                # Read actual contact force measured by sensors
                # sensor data is a flattened array, needs indexing by ID
                f_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "force_left")
                f_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "force_right")
                
                # force sensor returns 3 values (x, y, z), we usually care about resultant force or X-axis component
                # Note: sensor data indexing requires adr
                f_left_adr = model.sensor_adr[f_left_id]
                f_left_val = np.linalg.norm(data.sensordata[f_left_adr:f_left_adr+3])
                f_left_z = data.sensordata[f_left_adr+2]
                f_left_x = data.sensordata[f_left_adr]

                z_vel = data.qvel[2]
                status = "STABLE" if abs(z_vel) < 0.01 else "SLIPPING"
                
                print(f"Time: {data.time:.1f}s | Real Grip Force: Horizontal {f_left_z:.1f}N, Vertical {f_left_x:.1f}N | Obj Z-Vel: {z_vel:.4f} | {status}")

            viewer.sync()

            # Keep real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()