export interface Point {
  x: number;
  y: number;
}

export interface Path {
  points: Point[];
  length: number;
}

export interface VectorizedPaths {
  paths: Path[];
  totalPoints: number;
  totalPaths: number;
}

function traceContour(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  startX: number,
  startY: number,
  visited: Set<string>
): Point[] {
  const points: Point[] = [];
  const stack: Point[] = [{ x: startX, y: startY }];
  const directions = [
    { dx: 1, dy: 0 },
    { dx: 1, dy: 1 },
    { dx: 0, dy: 1 },
    { dx: -1, dy: 1 },
    { dx: -1, dy: 0 },
    { dx: -1, dy: -1 },
    { dx: 0, dy: -1 },
    { dx: 1, dy: -1 },
  ];

  while (stack.length > 0) {
    const current = stack.pop()!;
    const key = `${current.x},${current.y}`;

    if (visited.has(key)) continue;
    visited.add(key);

    const idx = (current.y * width + current.x) * 4;
    if (data[idx] > 127) {
      points.push(current);

      for (const dir of directions) {
        const nx = current.x + dir.dx;
        const ny = current.y + dir.dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const nkey = `${nx},${ny}`;
          if (!visited.has(nkey)) {
            const nidx = (ny * width + nx) * 4;
            if (data[nidx] > 127) {
              stack.push({ x: nx, y: ny });
            }
          }
        }
      }
    }
  }

  return points;
}

function simplifyPath(points: Point[], epsilon: number = 1.0): Point[] {
  if (points.length <= 2) return points;

  let dmax = 0;
  let index = 0;
  const end = points.length - 1;

  for (let i = 1; i < end; i++) {
    const d = perpendicularDistance(points[i], points[0], points[end]);
    if (d > dmax) {
      index = i;
      dmax = d;
    }
  }

  if (dmax > epsilon) {
    const left = simplifyPath(points.slice(0, index + 1), epsilon);
    const right = simplifyPath(points.slice(index), epsilon);
    return [...left.slice(0, -1), ...right];
  } else {
    return [points[0], points[end]];
  }
}

function perpendicularDistance(point: Point, lineStart: Point, lineEnd: Point): number {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;

  const mag = Math.sqrt(dx * dx + dy * dy);
  if (mag === 0) return Math.sqrt(Math.pow(point.x - lineStart.x, 2) + Math.pow(point.y - lineStart.y, 2));

  const u = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / (mag * mag);

  let closestX, closestY;
  if (u < 0) {
    closestX = lineStart.x;
    closestY = lineStart.y;
  } else if (u > 1) {
    closestX = lineEnd.x;
    closestY = lineEnd.y;
  } else {
    closestX = lineStart.x + u * dx;
    closestY = lineStart.y + u * dy;
  }

  return Math.sqrt(Math.pow(point.x - closestX, 2) + Math.pow(point.y - closestY, 2));
}

function calculatePathLength(points: Point[]): number {
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    length += Math.sqrt(dx * dx + dy * dy);
  }
  return length;
}

export function vectorizeMask(imageData: ImageData, minContourSize: number = 100): VectorizedPaths {
  const { width, height, data } = imageData;
  const visited = new Set<string>();
  const paths: Path[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const key = `${x},${y}`;

      if (data[idx] > 127 && !visited.has(key)) {
        const contourPoints = traceContour(data, width, height, x, y, visited);

        if (contourPoints.length >= minContourSize) {
          const simplified = simplifyPath(contourPoints, 1.0);
          paths.push({
            points: simplified,
            length: calculatePathLength(simplified)
          });
        }
      }
    }
  }

  paths.sort((a, b) => b.length - a.length);

  const totalPoints = paths.reduce((sum, path) => sum + path.points.length, 0);

  return {
    paths,
    totalPoints,
    totalPaths: paths.length
  };
}

export function createOverlayImage(
  originalImage: string,
  paths: Path[],
  width: number,
  height: number
): Promise<string> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) throw new Error('Could not get canvas context');

    canvas.width = width;
    canvas.height = height;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);

      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      paths.forEach((path) => {
        if (path.points.length < 2) return;

        ctx.beginPath();
        ctx.moveTo(path.points[0].x, path.points[0].y);

        for (let i = 1; i < path.points.length; i++) {
          ctx.lineTo(path.points[i].x, path.points[i].y);
        }

        ctx.stroke();
      });

      resolve(canvas.toDataURL('image/png'));
    };

    img.src = originalImage;
  });
}
