# CAPS test using new PUMI AIM

# Import pyCAPS class file
from pyCAPS import capsProblem
import math

# Initialize capsProblem object
myProblem = capsProblem()

# Load CSM file and build the geometry explicitly
myGeometry = myProblem.loadCAPS("motor.csm")

# myGeometry.setGeometryVal("rotor_rotation", 1.0)

myGeometry.buildGeometry()
myGeometry.saveGeometry("motor.egads")

# # Load EGADS file
# myGeometry = myProblem.loadCAPS("motor.egads")

# Working directory
workDir = "mesh"

# Load AFLR4 aim
# aflr4 = myProblem.loadAIM(aim = "egadsTessAIM",
aflr4 = myProblem.loadAIM(aim = "aflr4AIM",
                          analysisDir = workDir)

# Set project name so a mesh file is generated
aflr4.setAnalysisVal("Proj_Name", "surf_motor")

# Set AIM verbosity
# aflr4.setAnalysisVal("Mesh_Quiet_Flag", False)

# Set output grid format
aflr4.setAnalysisVal("Mesh_Format", "VTK")

# Set maximum and minimum edge lengths relative to capsMeshLength
# aflr4.setAnalysisVal("Mesh_Length_Factor", 1)
# aflr4.setAnalysisVal("Mesh_Elements", "Tri")
# aflr4.setAnalysisVal("Tess_Params", [0.00375, 0.025, 15])

aflr4.setAnalysisVal("max_scale", 0.005)
# aflr4.setAnalysisVal("min_scale", 0.1)
# aflr4.setAnalysisVal("ff_cdfr", 1.25)
# aflr4.setAnalysisVal("curv_factor", 0.5)
# aflr4.setAnalysisVal("Mesh_Gen_Input_String", "cdfr=1.5 curv_factor=1e16 min_nseg_loop=3 min_nseg=2")

# Run AIM pre-analysis
aflr4.preAnalysis()

#######################################
## AFRL4 was ran during preAnalysis ##
#######################################

# Run AIM post-analysis
aflr4.postAnalysis()

# Load TetGen aim with the surface mesh as the parent
tetgen = myProblem.loadAIM(aim='tetgenAIM', analysisDir= workDir, 
                           parents="aflr4AIM")
                        #    parents="egadsTessAIM")

# Set the tetgen analysis values
tetgen.setAnalysisVal('Proj_Name', 'vol_motor')
tetgen.setAnalysisVal('Mesh_Format', 'VTK')
# tetgen.setAnalysisVal("Tess_Params", [0.0025, 0.01, 15])

stator_od = myGeometry.getGeometryVal("stator_od")
stator_id = myGeometry.getGeometryVal("stator_id")
rotor_od = myGeometry.getGeometryVal("rotor_od")
rotor_id = myGeometry.getGeometryVal("rotor_id")
slot_depth = myGeometry.getGeometryVal("slot_depth")
tooth_width = myGeometry.getGeometryVal("tooth_width")
magnet_thickness = myGeometry.getGeometryVal("magnet_thickness")
heatsink_od = myGeometry.getGeometryVal("heatsink_od")
tooth_tip_thickness = myGeometry.getGeometryVal("tooth_tip_thickness")
tooth_tip_angle = myGeometry.getGeometryVal("tooth_tip_angle")
slot_radius = myGeometry.getGeometryVal("slot_radius")
stack_length = myGeometry.getGeometryVal("stack_length")
rotor_rotation = myGeometry.getGeometryVal("rotor_rotation")
num_slots = int(myGeometry.getGeometryVal("num_slots"))
num_magnets = int(myGeometry.getGeometryVal("num_magnets"))
mag_angle = 360.0 / num_magnets

# Sepecify a point in each region with the different ID's
regions = [
   ('farfield1', { 'id' : 1, 'seed' : [-2*stator_od/3, 0.0, 0.0] }),
   ('farfield2', { 'id' : 2, 'seed' : [2*stator_od/3, 0.0, 0.0] }),
   ('farfield3', { 'id' : 3, 'seed' : [0.0, 0.0, 0.0] }),
   ('stator', { 'id' : 4, 'seed' : [0.0, -(stator_od+3*stator_id)/8,  0.0] }),
   ('rotor', { 'id' : 5, 'seed' : [0.0, -(rotor_od+3*rotor_id)/8, 0.0] }),
   ('airgap', { 'id' : 6, 'seed' : [0.0, -(stator_id+6*(rotor_od/2+magnet_thickness))/8, 0.0] }),
]

for i in range(1, num_slots*4+1):
   r = (stator_id + stator_od) / 4
   dtheta = (60.0 / num_slots)*math.pi / 180.0
   theta = (((i-1)//4)*360.0 / num_slots)*math.pi / 180.0

   index = i % 4
   if (index == 1):
      x = r * math.cos(theta)
      y = r * math.sin(theta)

      z_offset = stack_length / 2
      z_inc = r * math.sin(dtheta);
      regions.append((f'coil{i}', {'id': regions[-1][1]['id']+1,
                                 'seed': [x, y, z_inc + z_offset]}))
   elif (index == 2):
      x = r * math.cos(theta+dtheta)
      y = r * math.sin(theta+dtheta)

      regions.append((f'coil{i}', {'id': regions[-1][1]['id']+1,
                                 'seed': [x, y, 0.0]}))
   elif (index == 3):
      x = r * math.cos(theta)
      y = r * math.sin(theta)

      z_offset = -stack_length / 2
      z_inc = -r * math.sin(dtheta);
      regions.append((f'coil{i}', {'id': regions[-1][1]['id']+1,
                                 'seed': [x, y, z_inc + z_offset]}))
   elif (index == 0):
      x = r * math.cos(theta-dtheta)
      y = r * math.sin(theta-dtheta)

      regions.append((f'coil{i}', {'id': regions[-1][1]['id']+1,
                                 'seed': [x, y, 0.0]}))
       
for i in range(1,num_magnets+1):
   r = rotor_od/2 + 0.5*magnet_thickness
   theta = (0.5*mag_angle + (i-1)*mag_angle)*math.pi / 180 + rotor_rotation
   x = r * math.cos(theta)
   y = r * math.sin(theta)
   regions.append((f'mag{i}', {'id': regions[-1][1]['id']+1, 'seed': [x, y, 0.0]}))

tetgen.setAnalysisVal('Regions', regions)

# Generate the volume mesh
tetgen.preAnalysis()
tetgen.postAnalysis()

pumi = myProblem.loadAIM(aim = "pumiAIM",
                           analysisDir = workDir,
                           parents="tetgenAIM")

pumi.setAnalysisVal("Proj_Name", "motor")

# Set AIM verbosity
# pumi.setAnalysisVal("Mesh_Quiet_Flag", False)

# Set output grid format
pumi.setAnalysisVal("Mesh_Format", "PUMI")
# pumi.setAnalysisVal("Mesh_Format", "VTK")
pumi.setAnalysisVal("Mesh_Order", 1)

# Run AIM pre-analysis
pumi.preAnalysis()

#####################################
## PUMI was ran during preAnalysis ##
#####################################

# Run AIM post-analysis
pumi.postAnalysis()

# Close CAPS
myProblem.closeCAPS()
