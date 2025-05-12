
import matplotlib.pyplot as plt

## INPUT PARAMETERS TO WORKFLOW AND UPDATE OF COMMAND FILE

# User parameters
heading = 0
xref = 25.875
yref = -62.817
hs = 2.5
tp = 10
xhook = 25.875
yhook = -44.817
SimLen = 2000

# Path to command file
file_path = r"C:\Users\marku\Documents\DT_prediction_model\DOLPHINN\SIMA_simulation\commandsModel.txt"

# Read the file content
with open(file_path, "r") as file:
    lines = file.readlines()

# Modify only the numbers in the second line
lines[1] = f"run task=Workflow1 workflow=Dynamic input=Heading={heading};Xref={xref};Yref={yref};Hs={hs};Tp={tp};Xhook={xhook};Yhook={yhook};SimLen={SimLen}\n"

# Write back the updated content
with open(file_path, "w") as file:
    file.writelines(lines)

# Read the file again and print its content
with open(file_path, "r") as file:
    print(file.read())


#############################################################3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Compute new point 30 meters ahead based on the heading
distance = 80.8/2
x_front = xref + distance * np.cos(np.radians(heading))
y_front = yref + distance * np.sin(np.radians(heading))
x_aft = xref - distance * np.cos(np.radians(heading))
y_aft = yref - distance * np.sin(np.radians(heading))

# Create figure and axis
fig, ax = plt.subplots()

# Set limits
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

# Ensure 1:1 aspect ratio
ax.set_aspect('equal', adjustable='box')

# Draw X and Y axis
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Add grid
ax.grid(True, linestyle='--', linewidth=0.5)

# Add labels
ax.set_xlabel("Y-axis")
ax.set_ylabel("X-axis")

# Add ticks for better visualization
ax.set_xticks(range(-100, 101, 10))
ax.set_yticks(range(-100, 101, 10))

# Floater Pontoons
pontoon_radius = 12.5 / 2  # Convert diameter to radius
pontoon_c = patches.Circle((0, 0), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2)
pontoon_1 = patches.Circle((-51.75, 0), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2)
pontoon_2 = patches.Circle((25.875, 44.817), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2)
pontoon_3 = patches.Circle((25.875, -44.817), pontoon_radius, edgecolor='purple', facecolor='none', linewidth=2)

ax.add_patch(pontoon_c)
ax.add_patch(pontoon_1)
ax.add_patch(pontoon_2)
ax.add_patch(pontoon_3)

ax.plot([0, -51.75], [0, 0], color='purple', linewidth=2, linestyle='-')
ax.plot([0, 25.875], [0, 44.817], color='purple', linewidth=2, linestyle='-')
ax.plot([0, 25.875], [0, -44.817], color='purple', linewidth=2, linestyle='-')

# HOOKUP POINTS AND DP SET-POINT
points = [(xref, yref), (xhook, yhook)]
for x, y in points:
    ax.scatter(x, y, color='red', s=50)  # Red points, size 50

# Draw line between reference point and new point
ax.plot([xref, xhook], [yref, yhook], color='green', linewidth=2, linestyle='-')
ax.plot([xref, x_front], [yref, y_front], color='blue', linewidth=2, linestyle='-')
ax.plot([xref, x_aft], [yref, y_aft], color='blue', linewidth=2, linestyle='-')


gangway_distance = np.sqrt((xhook - xref) ** 2 + (yhook - yref) ** 2)
# print(f"Gangway distance: {gangway_distance:.2f} meters")

# # Check if the gangway distance is within the desired range
# if 16 <= gangway_distance <= 21:
#     print("Gangway distance is within range.")
# else:
#     print("Gangway distance is out of range!")

# Show plot
plt.show()



