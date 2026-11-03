export const JOINT_DEFAULTS = {
  rSh: [0, 0, 0.12],
  rEl: [0, 0, 0],
  lSh: [0, 0, -0.12],
  lEl: [0, 0, 0],
};

export const pose = (overrides = {}) => ({
  ...JOINT_DEFAULTS,
  ...overrides,
});

export const REST = pose();

export const clonePose = (p) => JSON.parse(JSON.stringify(p));

export const lerpPose = (a, b, k) => {
  const out = {};
  for (const key of Object.keys(a)) {
    const av = a[key];
    const bv = b[key];
    out[key] = av.map((value, index) => value + (bv[index] - value) * k);
  }
  return out;
};

export const sampleClip = (clip, t) => {
  const keys = clip.keys;
  for (let i = 0; i < keys.length - 1; i += 1) {
    const current = keys[i];
    const next = keys[i + 1];
    if (t <= next.t) {
      const alpha = (t - current.t) / (next.t - current.t || 1);
      return lerpPose(current.pose, next.pose, Math.max(0, Math.min(1, alpha)));
    }
  }
  return clonePose(keys[keys.length - 1].pose);
};

export const SIGN_LIBRARY = {
  hello: {
    duration: 1.7,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.28, pose: pose({ rSh: [0.0, 0.0, 1.8], rEl: [0.0, 0.0, 0.9], lSh: [0.0, 0.0, -1.8], lEl: [0.0, 0.0, -0.9] }) },
      { t: 0.55, pose: pose({ rSh: [0.0, 0.0, 2.1], rEl: [0.0, 0.0, -0.6], lSh: [0.0, 0.0, -2.1], lEl: [0.0, 0.0, 0.6] }) },
      { t: 0.9, pose: pose({ rSh: [0.0, 0.0, 2.3], rEl: [0.0, 0.0, 0.9], lSh: [0.0, 0.0, -2.3], lEl: [0.0, 0.0, -0.9] }) },
      { t: 1.25, pose: pose({ rSh: [0.0, 0.0, 2.1], rEl: [0.0, 0.0, -0.7], lSh: [0.0, 0.0, -2.1], lEl: [0.0, 0.0, 0.7] }) },
      { t: 1.7, pose: pose() },
    ],
  },
  yes: {
    duration: 1.2,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.25, pose: pose({ rSh: [-0.8, 0.0, 0.25], rEl: [-1.1, 0.0, 0.0], lSh: [-0.2, 0.0, -0.1], lEl: [-0.35, 0.0, 0.0] }) },
      { t: 0.55, pose: pose({ rSh: [-1.3, 0.0, 0.2], rEl: [-1.9, 0.0, 0.0], lSh: [0.25, 0.0, -0.1], lEl: [0.25, 0.0, 0.0] }) },
      { t: 0.9, pose: pose({ rSh: [-0.9, 0.0, 0.25], rEl: [-1.2, 0.0, 0.0], lSh: [0.05, 0.0, -0.1], lEl: [0.05, 0.0, 0.0] }) },
      { t: 1.2, pose: pose() },
    ],
  },
  no: {
    duration: 1.2,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.25, pose: pose({ rSh: [0.9, 0.0, 0.25], rEl: [0.6, 0.0, 0.0], lSh: [0.2, 0.0, -0.25], lEl: [0.35, 0.0, 0.0] }) },
      { t: 0.6, pose: pose({ rSh: [1.6, 0.0, 0.7], rEl: [1.0, 0.0, 0.0], lSh: [-0.4, 0.0, -0.4], lEl: [0.15, 0.0, 0.0] }) },
      { t: 0.9, pose: pose({ rSh: [1.1, 0.0, 0.35], rEl: [0.9, 0.0, 0.0], lSh: [-0.15, 0.0, -0.25], lEl: [0.25, 0.0, 0.0] }) },
      { t: 1.2, pose: pose() },
    ],
  },
  thank_you: {
    duration: 1.5,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.35, pose: pose({ rSh: [-1.0, 0.0, 0.18], rEl: [-1.5, 0.0, 0.0], lSh: [-0.4, 0.0, -0.2], lEl: [-0.9, 0.0, 0.0] }) },
      { t: 0.8, pose: pose({ rSh: [-0.7, 0.0, 0.35], rEl: [-0.5, 0.0, 0.0], lSh: [-0.3, 0.0, 0.0], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 1.2, pose: pose({ rSh: [-0.55, 0.0, 0.2], rEl: [-0.25, 0.0, 0.0], lSh: [-0.25, 0.0, -0.15], lEl: [-0.25, 0.0, 0.0] }) },
      { t: 1.5, pose: pose() },
    ],
  },
  please: {
    duration: 1.35,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.28, pose: pose({ rSh: [-0.6, 0.0, 0.35], rEl: [-0.7, 0.0, 0.0], lSh: [-0.6, 0.0, -0.35], lEl: [-0.7, 0.0, 0.0] }) },
      { t: 0.7, pose: pose({ rSh: [-0.9, 0.0, 0.85], rEl: [-1.0, 0.0, 0.0], lSh: [-0.9, 0.0, -0.85], lEl: [-1.0, 0.0, 0.0] }) },
      { t: 1.0, pose: pose({ rSh: [-0.7, 0.0, 0.3], rEl: [-0.5, 0.0, 0.0], lSh: [-0.7, 0.0, -0.3], lEl: [-0.5, 0.0, 0.0] }) },
      { t: 1.35, pose: pose() },
    ],
  },
  sorry: {
    duration: 1.4,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.25, pose: pose({ rSh: [-0.8, 0.0, 0.4], rEl: [-0.6, 0.0, 0.0], lSh: [-0.8, 0.0, -0.4], lEl: [-0.6, 0.0, 0.0] }) },
      { t: 0.7, pose: pose({ rSh: [-1.1, 0.0, 0.8], rEl: [-1.2, 0.0, 0.0], lSh: [-1.1, 0.0, -0.8], lEl: [-1.2, 0.0, 0.0] }) },
      { t: 1.0, pose: pose({ rSh: [-0.7, 0.0, 0.5], rEl: [-0.4, 0.0, 0.0], lSh: [-0.7, 0.0, -0.5], lEl: [-0.4, 0.0, 0.0] }) },
      { t: 1.4, pose: pose() },
    ],
  },
  you: {
    duration: 0.9,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.3, pose: pose({ rSh: [-0.9, 0.0, 0.0], rEl: [-0.1, 0.0, 0.0], lSh: [0.0, 0.0, 0.0], lEl: [0.0, 0.0, 0.0] }) },
      { t: 0.6, pose: pose({ rSh: [-1.1, 0.0, 0.0], rEl: [-0.2, 0.0, 0.0], lSh: [0.0, 0.0, 0.0], lEl: [0.0, 0.0, 0.0] }) },
      { t: 0.9, pose: pose() },
    ],
  },
  me: {
    duration: 0.9,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.3, pose: pose({ rSh: [-0.5, 0.0, -0.8], rEl: [-1.3, 0.0, 0.0], lSh: [0.0, 0.0, 0.0], lEl: [0.0, 0.0, 0.0] }) },
      { t: 0.6, pose: pose({ rSh: [-0.6, 0.0, -1.0], rEl: [-1.8, 0.0, 0.0], lSh: [0.0, 0.0, 0.0], lEl: [0.0, 0.0, 0.0] }) },
      { t: 0.9, pose: pose() },
    ],
  },
  good: {
    duration: 1.1,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.35, pose: pose({ rSh: [-1.1, 0.0, 0.15], rEl: [-1.7, 0.0, 0.0], lSh: [-0.2, 0.0, -0.15], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 0.7, pose: pose({ rSh: [-0.35, 0.0, 0.1], rEl: [-0.4, 0.0, 0.0], lSh: [-0.2, 0.0, -0.1], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 1.1, pose: pose() },
    ],
  },
  bad: {
    duration: 1.1,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.35, pose: pose({ rSh: [-1.1, 0.0, 0.15], rEl: [-1.7, 0.0, 0.0], lSh: [0.2, 0.0, -0.15], lEl: [0.3, 0.0, 0.0] }) },
      { t: 0.7, pose: pose({ rSh: [0.25, 0.0, 0.1], rEl: [0.2, 0.0, 0.0], lSh: [0.3, 0.0, -0.15], lEl: [0.4, 0.0, 0.0] }) },
      { t: 1.1, pose: pose() },
    ],
  },
  help: {
    duration: 1.4,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.35, pose: pose({ rSh: [-0.4, 0.0, 0.18], rEl: [-0.2, 0.0, 0.0], lSh: [-0.4, 0.0, -0.18], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 0.8, pose: pose({ rSh: [-0.9, 0.0, 0.12], rEl: [-0.2, 0.0, 0.0], lSh: [-0.9, 0.0, -0.12], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 1.4, pose: pose() },
    ],
  },
  want: {
    duration: 1.15,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.35, pose: pose({ rSh: [-0.6, 0.0, 0.15], rEl: [-1.0, 0.0, 0.0], lSh: [-0.6, 0.0, -0.15], lEl: [-1.0, 0.0, 0.0] }) },
      { t: 0.75, pose: pose({ rSh: [-0.35, 0.0, 0.2], rEl: [-1.6, 0.0, 0.0], lSh: [-0.35, 0.0, -0.2], lEl: [-1.6, 0.0, 0.0] }) },
      { t: 1.15, pose: pose() },
    ],
  },
  how: {
    duration: 1.25,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.4, pose: pose({ rSh: [-0.7, 0.0, 0.3], rEl: [-1.4, 0.0, 0.0], lSh: [-0.7, 0.0, -0.3], lEl: [-1.4, 0.0, 0.0] }) },
      { t: 0.8, pose: pose({ rSh: [-0.5, 0.0, 0.65], rEl: [-1.0, 0.0, 0.0], lSh: [-0.5, 0.0, -0.65], lEl: [-1.0, 0.0, 0.0] }) },
      { t: 1.25, pose: pose() },
    ],
  },
  what: {
    duration: 1.0,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.35, pose: pose({ rSh: [-0.5, 0.0, 0.45], rEl: [-0.35, 0.0, 0.0], lSh: [-0.5, 0.0, -0.45], lEl: [-0.35, 0.0, 0.0] }) },
      { t: 0.65, pose: pose({ rSh: [-0.45, 0.0, -0.2], rEl: [-0.3, 0.0, 0.0], lSh: [-0.45, 0.0, 0.2], lEl: [-0.3, 0.0, 0.0] }) },
      { t: 1.0, pose: pose() },
    ],
  },
  fingerspell: {
    duration: 0.8,
    keys: [
      { t: 0.0, pose: pose() },
      { t: 0.16, pose: pose({ rSh: [-0.5, 0.0, 0.18], rEl: [-1.1, 0.3, 0.0], lSh: [-0.15, 0.0, -0.1], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 0.35, pose: pose({ rSh: [-0.5, 0.0, 0.18], rEl: [-1.1, -0.3, 0.0], lSh: [-0.15, 0.0, -0.1], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 0.56, pose: pose({ rSh: [-0.5, 0.0, 0.18], rEl: [-1.1, 0.3, 0.0], lSh: [-0.15, 0.0, -0.1], lEl: [-0.2, 0.0, 0.0] }) },
      { t: 0.8, pose: pose() },
    ],
  },
};

export const SIGN_NAMES = Object.keys(SIGN_LIBRARY);
