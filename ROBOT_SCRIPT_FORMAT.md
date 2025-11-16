# Robot Script Output Format

The application generates a JavaScript file (`04_robot_script.js`) with the following structure:

## Format Structure

```javascript
const PATHS = [
	// path 1
	[x1, x2, x3, ...],  // X coordinates for path 1
	[y1, y2, y3, ...],  // Y coordinates for path 1
	// path 2
	[x1, x2, x3, ...],  // X coordinates for path 2
	[y1, y2, y3, ...],  // Y coordinates for path 2
	// ... more paths
];

var x0 = 0.0;           // Origin offset X in mm
var y0 = 0.0;           // Origin offset Y in mm
var z0 = 5.0;           // Z-height in mm
var dx = 0;             // Reserved for future use
var dy = 0;             // Reserved for future use
var dz = 10;            // Z-lift height in mm
var m_v_move = 80;      // Movement speed (4x draw speed)
var m_v_draw = 20;      // Drawing/welding speed
var m_a = 1000;         // Acceleration

for (var i = 0; i < PATHS.length; i += 2) {
	// Y axis of image → X axis of robot
	const path_x = PATHS[i+1];
	// X axis of image → Y axis of robot
	const path_y = PATHS[i];

	if (path_x.length < 3) {
		continue;
	}

	// Move to start position (lifted)
	moveLinear('tcp', {
		x:x0+path_x[0],
		y:y0+path_y[0],
		z:z0+dz,
		rx:180.00,
		ry:0.00,
		rz:90.00
	}, m_v_move, m_a, {'precisely':false});

	console.log((x0+path_x[0]) + ', ' + (y0+path_y[0]));

	// Draw the path
	for (var j = 0; j < path_x.length; j++) {
		moveLinear('tcp', {
			x:x0+path_x[j],
			y:y0+path_y[j],
			z:z0,
			rx:180.00,
			ry:0.00,
			rz:90.00
		}, m_v_draw, m_a, {'precisely':false});
	}

	// Lift at end
	moveLinear('tcp', {
		x:x0+path_x[path_x.length-1],
		y:y0+path_y[path_y.length-1],
		z:z0+dz,
		rx:180.00,
		ry:0.00,
		rz:90.00
	}, m_v_draw, m_a, {'precisely':false});
}
```

## Key Features

### Path Data Structure
- Paths are stored as pairs of arrays: X coordinates followed by Y coordinates
- Each path occupies 2 array positions in PATHS (indices i and i+1)
- Coordinates are integer pixel values from the 256×384 image

### Coordinate System
- **Image X axis** maps to **Robot Y axis**
- **Image Y axis** maps to **Robot X axis**
- This accounts for the typical robot coordinate system orientation

### Movement Logic
1. **Approach**: Move to start position with Z lifted (z0 + dz)
2. **Draw**: Execute welding path at working height (z0)
3. **Retract**: Lift Z at the end of path (z0 + dz)

### Filtering
- Paths with fewer than 3 points are skipped (`if (path_x.length < 3)`)
- This removes noise and ensures meaningful welding paths

## Configuration Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `x0` | X origin offset (mm) | User configured |
| `y0` | Y origin offset (mm) | User configured |
| `z0` | Working Z-height (mm) | User configured |
| `dz` | Z-lift amount (mm) | 10 |
| `m_v_move` | Travel speed | 4× draw speed |
| `m_v_draw` | Welding speed | User configured |
| `m_a` | Acceleration | 1000 |

## Example Output

```javascript
const PATHS = [
	// path 1
	[39, 37, 33, 26, 22, 20, 20, 21, 21, 26, 44],
	[92, 90, 85, 78, 74, 66, 61, 56, 47, 43, 39],
	// path 2
	[151, 150, 149, 148],
	[44, 43, 42, 41],
	// path 3
	[36, 37, 40, 44, 70, 85, 98],
	[54, 50, 47, 43, 39, 37, 37],
];
var x0 = -825.0;
var y0 = -115.0;
var z0 = -363.7;
var dx = 0;
var dy = 0;
var dz = 10;
var m_v_move = 200;
var m_v_draw = 50;
var m_a = 1000;

// ... robot control loop follows
```

## Integration Notes

This format is designed for direct execution on robotic systems that support:
- JavaScript-based motion control
- `moveLinear()` function for TCP movement
- Object-based coordinate specification
- Asynchronous execution

The script can be loaded directly into compatible robot controllers or adapted for other systems by modifying the `moveLinear()` calls to match your robot's API.
