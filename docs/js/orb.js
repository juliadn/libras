// Siri-style orb drawn on a canvas. Colored light blobs orbit inside a glass
// sphere; `energy` (0..1) sets their speed and size, `pulse()` makes it bloom.

const BLOBS = [
  { color: [0, 194, 255], r: 0.62, speed: 0.9, orbit: 0.30, phase: 0.0 },
  { color: [122, 92, 255], r: 0.70, speed: -0.7, orbit: 0.26, phase: 2.1 },
  { color: [255, 45, 149], r: 0.58, speed: 1.15, orbit: 0.33, phase: 4.0 },
  { color: [255, 159, 10], r: 0.42, speed: -1.3, orbit: 0.28, phase: 5.2 },
  { color: [52, 230, 200], r: 0.40, speed: 0.6, orbit: 0.22, phase: 1.2 },
];

export class Orb {
  constructor(canvas, glow, wrap) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.glow = glow;
    this.wrap = wrap;
    this.energy = 0.15;
    this.target = 0.15;
    this.bloom = 0;
    this.phase = 0;
    this.last = performance.now();
    this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = Math.round(this.c.clientWidth * dpr) || 220;
    if (this.c.width !== size) { this.c.width = size; this.c.height = size; }
  }

  setEnergy(e) { this.target = e; }

  pulse(strength = 1) { this.bloom = Math.max(this.bloom, strength); }

  shake() {
    this.wrap.classList.remove("shake");
    void this.wrap.offsetWidth;
    this.wrap.classList.add("shake");
  }

  draw(now) {
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.energy += (this.target - this.energy) * Math.min(1, dt * 3);
    this.bloom *= Math.exp(-dt * 2.2);
    const e = Math.min(1, this.energy + this.bloom * 0.8);
    this.phase += dt * (this.reduced ? 0.15 : 0.35 + e * 1.9);

    const ctx = this.ctx, S = this.c.width, R = S / 2;
    const breathe = this.reduced ? 0 : Math.sin(now / 1400) * 0.015;
    const scale = 0.86 + e * 0.08 + this.bloom * 0.06 + breathe;
    const rr = R * scale;

    ctx.clearRect(0, 0, S, S);
    ctx.save();
    ctx.beginPath();
    ctx.arc(R, R, rr, 0, Math.PI * 2);
    ctx.clip();

    // deep glass base
    const base = ctx.createRadialGradient(R, R * 0.9, rr * 0.1, R, R, rr);
    base.addColorStop(0, "#1a1440");
    base.addColorStop(0.75, "#07051a");
    base.addColorStop(1, "#000000");
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, S, S);

    // light blobs
    ctx.globalCompositeOperation = "lighter";
    for (const b of BLOBS) {
      const a = this.phase * b.speed + b.phase;
      const orbit = rr * b.orbit * (0.75 + e * 0.6);
      const x = R + Math.cos(a) * orbit;
      const y = R + Math.sin(a * 1.3) * orbit * 0.85;
      const br = rr * b.r * (0.8 + e * 0.35);
      const g = ctx.createRadialGradient(x, y, 0, x, y, br);
      const alpha = 0.5 + e * 0.4;
      g.addColorStop(0, `rgba(${b.color},${alpha})`);
      g.addColorStop(0.55, `rgba(${b.color},${alpha * 0.35})`);
      g.addColorStop(1, `rgba(${b.color},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, br, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";

    // glass highlight + rim
    const hl = ctx.createRadialGradient(R - rr * 0.35, R - rr * 0.45, 0, R - rr * 0.35, R - rr * 0.45, rr * 0.75);
    hl.addColorStop(0, "rgba(255,255,255,0.35)");
    hl.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = hl;
    ctx.fillRect(0, 0, S, S);
    const rim = ctx.createRadialGradient(R, R, rr * 0.82, R, R, rr);
    rim.addColorStop(0, "rgba(255,255,255,0)");
    rim.addColorStop(1, "rgba(255,255,255,0.22)");
    ctx.fillStyle = rim;
    ctx.fillRect(0, 0, S, S);
    ctx.restore();

    this.glow.style.setProperty("--glow", (0.18 + e * 0.55).toFixed(3));
    this.glow.style.transform = `scale(${(0.72 + e * 0.35 + this.bloom * 0.15).toFixed(3)}) rotate(${(this.phase * 20) % 360}deg)`;
  }
}
