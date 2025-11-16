import { Path, Point } from './vectorization';

export interface CoordinateConfig {
  mmPerPixel: number;
  originOffsetX: number;
  originOffsetY: number;
  zHeight: number;
}

export interface RobotScriptParams {
  paths: Path[];
  speed: number;
  config: CoordinateConfig;
  imageHeight: number;
}

function pixelToRobotCoords(point: Point, config: CoordinateConfig, imageHeight: number): Point {
  return {
    x: point.x * config.mmPerPixel + config.originOffsetX,
    y: (imageHeight - point.y) * config.mmPerPixel + config.originOffsetY
  };
}

export function generateABBScript(params: RobotScriptParams): string {
  const { paths, speed, config, imageHeight } = params;
  const lines: string[] = [];

  lines.push('const PATHS = [');

  paths.forEach((path, pathIndex) => {
    lines.push(`\t// path ${pathIndex + 1}`);

    const xCoords = path.points.map(p => Math.round(p.x)).join(', ');
    lines.push(`\t[${xCoords}],`);

    const yCoords = path.points.map(p => Math.round(p.y)).join(', ');
    lines.push(`\t[${yCoords}],`);
  });

  lines.push('];');
  lines.push(`var x0 = ${config.originOffsetX.toFixed(1)};`);
  lines.push(`var y0 = ${config.originOffsetY.toFixed(1)};`);
  lines.push(`var z0 = ${config.zHeight.toFixed(1)};`);
  lines.push('var dx = 0;');
  lines.push('var dy = 0;');
  lines.push('var dz = 10;');
  lines.push(`var m_v_move = ${speed * 4};`);
  lines.push(`var m_v_draw = ${speed};`);
  lines.push('var m_a = 1000;');
  lines.push('');
  lines.push('for (var i = 0; i < PATHS.length; i += 2) {');
  lines.push('\t// Oś Y obrazu rysujemy wzdłuż osi X robota.');
  lines.push('\tconst path_x = PATHS[i+1];');
  lines.push('\t// Oś X obrazu rysujemy wzdłuż osi Y robota.');
  lines.push('\tconst path_y = PATHS[i];');
  lines.push('\tif (path_x.length < 3) {');
  lines.push('\t\tcontinue;');
  lines.push('\t}');
  lines.push('\tmoveLinear(\'tcp\', {x:x0+path_x[0], y:y0+path_y[0], z:z0+dz, rx:180.00, ry:0.00, rz:90.00}, m_v_move, m_a, {\'precisely\':false});');
  lines.push('\tconsole.log((x0+path_x[0]) + \', \' + (y0+path_y[0]));');
  lines.push('\tfor (var j = 0; j < path_x.length; j++) {');
  lines.push('\t\tmoveLinear(\'tcp\', {x:x0+path_x[j], y:y0+path_y[j], z:z0, rx:180.00, ry:0.00, rz:90.00}, m_v_draw, m_a, {\'precisely\':false});');
  lines.push('\t}');
  lines.push('\tmoveLinear(\'tcp\', {x:x0+path_x[path_x.length-1], y:y0+path_y[path_y.length-1], z:z0+dz, rx:180.00, ry:0.00, rz:90.00}, m_v_draw, m_a, {\'precisely\':false});');
  lines.push('}');

  return lines.join('\n');
}

export function generateGCode(params: RobotScriptParams): string {
  const { paths, speed, config, imageHeight } = params;
  const lines: string[] = [];

  lines.push('; G-code for CNC Plasma Cutter');
  lines.push(`; Generated: ${new Date().toISOString()}`);
  lines.push(`; Total paths: ${paths.length}`);
  lines.push(`; Feed rate: ${speed * 60} mm/min`);
  lines.push('');
  lines.push('G21 ; Set units to millimeters');
  lines.push('G90 ; Absolute positioning');
  lines.push('G17 ; XY plane selection');
  lines.push(`F${speed * 60} ; Set feed rate`);
  lines.push('');

  paths.forEach((path, pathIndex) => {
    lines.push(`; Path ${pathIndex + 1}`);

    path.points.forEach((point, pointIndex) => {
      const robotCoord = pixelToRobotCoords(point, config, imageHeight);

      if (pointIndex === 0) {
        lines.push(`G0 Z${(config.zHeight + 5).toFixed(2)} ; Lift up`);
        lines.push(`G0 X${robotCoord.x.toFixed(2)} Y${robotCoord.y.toFixed(2)} ; Rapid to start`);
        lines.push(`G1 Z${config.zHeight.toFixed(2)} ; Lower to cut height`);
        lines.push('M3 ; Start plasma');
      } else {
        lines.push(`G1 X${robotCoord.x.toFixed(2)} Y${robotCoord.y.toFixed(2)} ; Cut`);
      }
    });

    lines.push('M5 ; Stop plasma');
    lines.push('');
  });

  lines.push('G0 Z10 ; Lift up');
  lines.push('M2 ; End program');

  return lines.join('\n');
}

export function generateParametersFile(params: RobotScriptParams): string {
  const { paths, speed, config } = params;

  const lines: string[] = [];
  lines.push('WELDING PATH GENERATION PARAMETERS');
  lines.push('=' .repeat(50));
  lines.push('');
  lines.push(`Generation Date: ${new Date().toISOString()}`);
  lines.push('');
  lines.push('IMAGE SETTINGS:');
  lines.push(`  Resolution: 256 x 384 pixels`);
  lines.push(`  Aspect Ratio: 2:3`);
  lines.push('');
  lines.push('COORDINATE TRANSFORMATION:');
  lines.push(`  mm per pixel: ${config.mmPerPixel}`);
  lines.push(`  Origin Offset X: ${config.originOffsetX} mm`);
  lines.push(`  Origin Offset Y: ${config.originOffsetY} mm`);
  lines.push(`  Z-Height: ${config.zHeight} mm`);
  lines.push('');
  lines.push('WELDING PARAMETERS:');
  lines.push(`  Speed: ${speed} mm/s (${speed * 60} mm/min)`);
  lines.push(`  Feed Rate: ${speed * 60} mm/min`);
  lines.push('');
  lines.push('PATH STATISTICS:');
  lines.push(`  Total Paths: ${paths.length}`);
  lines.push(`  Total Points: ${paths.reduce((sum, p) => sum + p.points.length, 0)}`);

  let totalLength = 0;
  paths.forEach((path, idx) => {
    lines.push(`  Path ${idx + 1}: ${path.points.length} points, ${path.length.toFixed(2)}px (${(path.length * config.mmPerPixel).toFixed(2)}mm)`);
    totalLength += path.length * config.mmPerPixel;
  });

  lines.push('');
  lines.push(`ESTIMATED VALUES:`);
  lines.push(`  Total Path Length: ${totalLength.toFixed(2)} mm`);
  lines.push(`  Estimated Time: ${(totalLength / speed).toFixed(2)} seconds`);

  return lines.join('\n');
}
