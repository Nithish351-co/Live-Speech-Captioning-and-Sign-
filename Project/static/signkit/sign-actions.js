const ALIASES = Object.freeze({
  HI: 'HELLO',
  BYE: 'HELLO',
  GOODBYE: 'HELLO',
});

export const SIGN_ANIMATIONS = Object.freeze({
  HELLO: 'Wave',
  HI: 'Wave',
  BYE: 'Wave',
  GOODBYE: 'Wave',
  YES: 'Yes',
  NO: 'No',
  NOT: 'No',
});

const CLIP_ANIMATIONS = Object.freeze({
  hello: 'Wave',
  yes: 'Yes',
  no: 'No',
});

const SIGNKIT_WORD_CLIPS = Object.freeze({
  signkit_home: 'HOME',
  signkit_person: 'PERSON',
  signkit_time: 'TIME',
  you: 'YOU',
});

export function resolveSignAnimation(gloss, clip) {
  const key = String(gloss || '').toUpperCase();
  const clipKey = String(clip || '').toLowerCase();
  return SIGN_ANIMATIONS[ALIASES[key] || key] || CLIP_ANIMATIONS[clipKey] || null;
}

export function resolveSignKitWord(gloss, clip) {
  const key = String(gloss || '').toUpperCase();
  const clipKey = String(clip || '').toLowerCase();
  return SIGNKIT_WORD_CLIPS[clipKey] || SIGNKIT_WORD_CLIPS[`signkit_${key.toLowerCase()}`] || null;
}
