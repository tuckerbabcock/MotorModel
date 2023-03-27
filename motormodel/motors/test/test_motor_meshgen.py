import pyCAPS

# Initialize capsProblem object
# project_name = "mesh_motor2D_true"
# csm_file = "../model/motor2D_true.csm"
project_name = "test_motor"
csm_file = "model/test_motor.csm"
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
    surface_mesh.input.Mesh_Sizing = {
        "air_gap": {"tessParams": [0.00075, 0.0, 0.0]},
        "tooth": {"tessParams": [0.00075, 0.0, 0.0]},
        "winding": {"tessParams": [0.00075, 0.0, 0.0]},
        "magnets": {"tessParams": [0.00075, 0.0, 0.0]},
        "rotor": {"tessParams": [0.00075, 0.0, 0.0]},
    }
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

pumi = myProblem.analysis.create(aim = "pumiAIM")
# pumi.input["Volume_Mesh"].link(volume_mesh.output["Volume_Mesh"])
pumi.input["Surface_Mesh"].link(surface_mesh.output["Surface_Mesh"])

# Set project name so a mesh file is generated
pumi.input.Proj_Name = project_name

# Set output grid format
pumi.input.Mesh_Format = "PUMI"
# pumi.input.Mesh_Format = "VTK"

# Set mesh element order
pumi.input.Mesh_Order = 1

# Convert to pumi mesh
pumi.runAnalysis()
