import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/research.ts", import.meta.url), "utf8");
const javascript = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 } }).outputText;
const { pointAt, peakScore, clipBounds } = await import(`data:text/javascript;base64,${Buffer.from(javascript).toString("base64")}`);
const points = [
  { time_s: 1, score: .9, valid: true, audio_energy: 10 },
  { time_s: 1.04, score: .99, valid: false, audio_energy: 0 },
  { time_s: 1.08, score: .6, valid: true, audio_energy: 8 },
];

test("playhead never borrows a score from a distant analysed window", () => {
  assert.equal(pointAt(points, 0), undefined);
  assert.equal(pointAt(points, 2), undefined);
  assert.equal(pointAt([], 1), undefined);
  assert.equal(pointAt(points, 1.081), points[2]);
});
test("invalid windows stay invalid and are excluded from peak evidence", () => {
  assert.equal(pointAt(points, 1.04).valid, false);
  assert.equal(peakScore(points, [1, 1.08]), .9);
  assert.equal(peakScore(points, [1.03, 1.05]), undefined);
});
test("clip context is bounded by source start and duration", () => {
  assert.deepEqual(clipBounds([.2, 1.4], .75, 2), [0, 2]);
  assert.deepEqual(clipBounds([2, 3], 0, 10), [2, 3]);
  assert.deepEqual(clipBounds([2, 3], .75, 0), [1.25, 3.75]);
});
