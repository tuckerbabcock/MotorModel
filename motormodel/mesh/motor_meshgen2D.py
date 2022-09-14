import pyCAPS
import math

# Initialize capsProblem object
project_name = "mesh_motor2D"
csm_file = "../model/motor2D.csm"
myProblem = pyCAPS.Problem(problemName=project_name,
                           capsFile=csm_file) 

# Load surface mesh aim
surf_mesh_aim = "egadsTessAIM"
# surf_mesh_aim = "aflr4AIM"
surface_mesh = myProblem.analysis.create(aim = surf_mesh_aim)

# Set project name so a mesh file is generated
surface_mesh.input.Proj_Name = "surf_"+project_name;

# Set AIM verbosity
surface_mesh.input.Mesh_Quiet_Flag = False

# Set output grid format
surface_mesh.input.Mesh_Format = "VTK"
if surf_mesh_aim == "egadsTessAIM":
    pass
elif surf_mesh_aim == "aflr4AIM":
    surface_mesh.input.Mesh_Sizing = {
        "air_core": {"scaleFactor" : 50},
        "air_gap": {"scaleFactor" : 0.1},
        "tooth": {"scaleFactor" : 0.5},
        "winding": {"scaleFactor" : 0.5},
        "magnets": {"scaleFactor" : 0.25},
        "rotor": {"scaleFactor" : 0.5},
        "motor": {"scaleFactor": 10}
    }

# Generate the surface mesh
surface_mesh.runAnalysis()

# Load volume mesh aim
vol_mesh_aim = "tetgenAIM"
volume_mesh = myProblem.analysis.create(aim = vol_mesh_aim)

# Link the surface mesh
volume_mesh.input["Surface_Mesh"].link(surface_mesh.output["Surface_Mesh"])

# Set the volume_mesh analysis values
# Set project name so a mesh file is generated
volume_mesh.input.Proj_Name = "vol_"+project_name

# Set output grid format
volume_mesh.input.Mesh_Format = "VTK"
# volume_mesh.input.Mesh_Format = "AFLR3"

geom = myProblem.geometry

stator_od = geom.despmtr["stator_od"].value
stator_id = geom.despmtr["stator_id"].value
rotor_od = geom.despmtr["rotor_od"].value
rotor_id = geom.despmtr["rotor_id"].value
slot_depth = geom.despmtr["slot_depth"].value
tooth_width = geom.despmtr["tooth_width"].value
magnet_thickness = geom.despmtr["magnet_thickness"].value
tooth_tip_thickness = geom.despmtr["tooth_tip_thickness"].value
tooth_tip_angle = geom.despmtr["tooth_tip_angle"].value
slot_radius = geom.despmtr["slot_radius"].value

heatsink_thickness = geom.cfgpmtr["heatsink_thickness"].value
shaft_thickness = geom.cfgpmtr["shaft_thickness"].value

num_slots = int(geom.cfgpmtr["num_slots"].value)
num_magnets = int(geom.conpmtr["num_magnets"].value) * int(geom.conpmtr["magnet_divisions"].value)
mag_angle = 360.0 / num_magnets
rotor_rotation = geom.conpmtr["rotor_rotation"].value

# Sepecify a point in each region with the different IDs
regions = {
    "stator": {"id": 1, "seed": [(stator_od+3*stator_id)/8,  0.0, 0.0]},
    "rotor": {"id": 2, "seed": [(rotor_od+3*rotor_id)/8, 0.0, 0.0]},
    "airgap": {"id": 3, "seed": [(stator_id+6*(rotor_od/2+magnet_thickness))/8, 0.0, 0.0]},
    # "airecore": {"id": 4, "seed": [0.0, 0.0, 0.0]}
    # "shaft": {"id": 4, "seed": [rotor_id/2 - shaft_thickness/2, 0.0, 0.0]},
    # "heatsink": {"id": 5, "seed": [stator_od/2 + heatsink_thickness/2, 0.0, 0.0]}
}

print("Regions:", regions)

for i in range(1, num_slots*2+1):
    r = (stator_id + stator_od) / 4
    dtheta = (60.0 / num_slots) * math.pi / 180.0
    theta = (((i-1)//2)*360.0 / num_slots) * math.pi / 180.0

    region_name = f"coil{i}" 

    index = i % 2
    if (index == 0):
        x = r * math.cos(theta+dtheta)
        y = r * math.sin(theta+dtheta)
        regions[region_name] = {"id": len(regions)+1,
                                "seed": [x, y, 0.0]}
    elif (index == 1):
        x = r * math.cos(theta-dtheta)
        y = r * math.sin(theta-dtheta)
        regions[region_name] = {"id": len(regions)+1,
                                "seed": [x, y, 0.0]}

for i in range(1,num_magnets+1):
    r = rotor_od/2 + 0.5*magnet_thickness
    theta = (0.5*mag_angle + (i-1)*mag_angle + rotor_rotation) * math.pi / 180
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    region_name = f"mag{i}" 
    regions[region_name] = {"id": len(regions)+1,
                            "seed": [x, y, 0.0]}

volume_mesh.input.Regions = regions
volume_mesh.input.Preserve_Surf_Mesh = True

# Generate the volume mesh
volume_mesh.runAnalysis()

pumi = myProblem.analysis.create(aim = "pumiAIM")
pumi.input["Volume_Mesh"].link(volume_mesh.output["Volume_Mesh"])
# pumi.input["Mesh"].link(surface_mesh.output["Surface_Mesh"])

# Set project name so a mesh file is generated
pumi.input.Proj_Name = project_name

# Set output grid format
pumi.input.Mesh_Format = "PUMI"
# pumi.input.Mesh_Format = "VTK"

# Set mesh element order
pumi.input.Mesh_Order = 1

# Convert to pumi mesh
pumi.runAnalysis()
