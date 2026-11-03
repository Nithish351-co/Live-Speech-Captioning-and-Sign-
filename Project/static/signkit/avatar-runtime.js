import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FINGER_SPELLING_LETTER_SECONDS, LETTER_HANDSHAPES } from './finger-spelling.js';

const FINGER_JOINTS = [
  ['IndexR', 0],
  ['Index2R', 0],
  ['Middle1R', 1],
  ['Middle2R', 1],
  ['Ring1R', 2],
  ['Ring2R', 2],
  ['ThumbR', 3],
  ['Thumb2R', 3],
  ['LowerArmR', 4],
  ['UpperArmR', 5],
  ['ShoulderR', 6],
];
const FINGER_BONES_TO_LENGTHEN = [
  'IndexR', 'Index2R',
  'Middle1R', 'Middle2R',
  'Ring1R', 'Ring2R',
  'ThumbR', 'Thumb2R',
];
const FINGER_LENGTH_SCALE = 1.5;
const WRIST_TILT_RADIANS = THREE.MathUtils.degToRad(140);
const WRIST_ROOT_NAMES = ['Palm1R', 'Palm2R', 'Palm3R'];
const FINGER_COLOR = 0x328cff;
const HAND_COLOR = 0xca9337;
const PHONE_AVATAR_YAW = -0.35;
// Raised arm pose adapted from the avatar's Wave clip, with additional upper-arm lift.
const SIGNING_ARM_ROTATIONS = Object.freeze({
  ShoulderR: [-0.038296, 0.0386762, 0.7095304, 0.7025697],
  UpperArmR: [0.0315815, 0.8065547, 0.0169676, 0.5900714],
});
const SIGNKIT_WORD_POSES = Object.freeze({
  HOME: {
    leftHandshape: 'B',
    rightHandshape: 'B',
    joints: {
      ShoulderL: [-0.1, 0, -0.55],
      UpperArmL: [-0.45, 0, -0.2],
      LowerArmL: [-0.8, 0, 0],
      ShoulderR: [-0.1, 0, 0.55],
      UpperArmR: [-0.45, 0, 0.2],
      LowerArmR: [-0.8, 0, 0],
    },
  },
  PERSON: {
    rightHandshape: 'D',
    joints: {
      ShoulderR: [-0.5, 0, 0.15],
      UpperArmR: [-0.45, 0, 0.1],
      LowerArmR: [-0.9, 0, 0],
    },
  },
  TIME: {
    leftHandshape: 'S',
    rightHandshape: 'D',
    joints: {
      ShoulderL: [-0.55, 0, -0.15],
      UpperArmL: [-0.35, 0, -0.15],
      LowerArmL: [-0.65, 0, 0],
      ShoulderR: [-0.55, 0, 0.15],
      UpperArmR: [-0.35, 0, 0.15],
      LowerArmR: [-0.65, 0, 0],
    },
  },
  YOU: {
    rightHandshape: 'D',
    joints: {
      ShoulderR: [-0.45, 0, 0.05],
      UpperArmR: [-0.25, 0, 0],
      LowerArmR: [-0.75, 0, 0],
    },
  },
});

export async function createSignAvatar(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  container.replaceChildren(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(-3.5, 2.6, 7);
  camera.lookAt(0, 2, 0);
  const idleCameraPosition = camera.position.clone();
  const idleCameraTarget = new THREE.Vector3(0, 2, 0);
  const signingCameraPosition = new THREE.Vector3(-2.6, 3.45, 4.3);
  const signingCameraTarget = new THREE.Vector3(-1.9, 2.95, 0.4);
  const cameraTarget = idleCameraTarget.clone();

  scene.add(new THREE.HemisphereLight(0xdcecff, 0x26304a, 2.1));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
  keyLight.position.set(2.5, 4, 3.5);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x72d8ff, 1.1);
  fillLight.position.set(-3, 2, 1);
  scene.add(fillLight);

  const avatar = await loadSignAvatar(scene);
  avatar.model.rotation.y = PHONE_AVATAR_YAW;

  const resize = () => {
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const delta = Math.min(clock.getDelta(), 0.1);
    avatar.update(delta, camera.position);
    const focus = avatar.fingerSpellingActive || avatar.wordSigningActive ? 1 : 0;
    const cameraEase = 1 - Math.exp(-delta * 6);
    camera.position.lerp(focus ? signingCameraPosition : idleCameraPosition, cameraEase);
    cameraTarget.lerp(focus ? signingCameraTarget : idleCameraTarget, cameraEase);
    camera.lookAt(cameraTarget);
    renderer.render(scene, camera);
  });

  return avatar;
}

export async function loadSignAvatar(scene) {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(new URL('./RobotExpressive.glb', import.meta.url).href);
  gltf.scene.traverse(node => {
    if (node.isSkinnedMesh) node.frustumCulled = false;
    if (node.isSkinnedMesh && /^Hand[LR]_1$/.test(node.name)) {
      const handMaterial = node.material.clone();
      handMaterial.color.setHex(HAND_COLOR);
      handMaterial.needsUpdate = true;
      node.material = handMaterial;
    }
    if (node.isSkinnedMesh && /^Hand[LR]_2$/.test(node.name)) {
      node.geometry = node.geometry.clone();
      const position = node.geometry.getAttribute('position');
      const skinIndex = node.geometry.getAttribute('skinIndex');
      const skinWeight = node.geometry.getAttribute('skinWeight');
      const vertexColors = new Float32Array(position.count * 3);
      const handColor = new THREE.Color(HAND_COLOR);
      const fingerColor = new THREE.Color(FINGER_COLOR);
      for (let vertex = 0; vertex < position.count; vertex += 1) {
        let dominantBone = -1;
        let dominantWeight = -1;
        for (let influence = 0; influence < 4; influence += 1) {
          const weight = skinWeight.getComponent(vertex, influence);
          if (weight > dominantWeight) {
            dominantWeight = weight;
            dominantBone = skinIndex.getComponent(vertex, influence);
          }
        }
        const boneName = node.skeleton.bones[dominantBone]?.name ?? '';
        const isFinger = /^(Index|Middle|Ring|Thumb)(2)?[LR]$/.test(boneName);
        (isFinger ? fingerColor : handColor).toArray(vertexColors, vertex * 3);
      }
      node.geometry.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3));
      const fingerMaterial = node.material.clone();
      fingerMaterial.color.setHex(0xffffff);
      fingerMaterial.vertexColors = true;
      fingerMaterial.needsUpdate = true;
      node.material = fingerMaterial;
    }
  });
  scene.add(gltf.scene);

  const mixer = new THREE.AnimationMixer(gltf.scene);
  const actions = new Map();
  for (const clip of gltf.animations) {
    actions.set(clip.name.toLowerCase().replace(/[^a-z]/g, ''), mixer.clipAction(clip));
  }

  const idle = actions.get('idle');
  if (!idle) throw new Error('The avatar model is missing its Idle animation.');
  idle.play();

  const controlledBones = [];
  gltf.scene.traverse(node => {
    if (node.isBone && FINGER_JOINTS.some(([name]) => name === node.name)) {
      controlledBones.push({ node, base: node.quaternion.clone(), baseScale: node.scale.clone() });
    }
  });
  const wristRoots = WRIST_ROOT_NAMES.map(name => {
    const node = gltf.scene.getObjectByName(name);
    if (!node) throw new Error(`The avatar model is missing the "${name}" articulation joint.`);
    return { node, base: node.quaternion.clone() };
  });
  for (const [name] of FINGER_JOINTS) {
    if (!controlledBones.some(({ node }) => node.name === name)) {
      throw new Error(`The avatar model is missing the "${name}" articulation joint.`);
    }
  }
  for (const name of FINGER_BONES_TO_LENGTHEN) {
    const bone = controlledBones.find(({ node }) => node.name === name);
    if (bone) bone.node.scale.y = bone.baseScale.y * FINGER_LENGTH_SCALE;
  }

  const jointIndex = new Map(FINGER_JOINTS.map(([name], index) => [name, index]));
  const neutralPose = controlledBones.map(({ base }) => base.clone());
  const wordBones = [];
  const wordJointNames = new Set([
    'ShoulderL', 'UpperArmL', 'LowerArmL',
    'ShoulderR', 'UpperArmR', 'LowerArmR',
    ...FINGER_JOINTS.slice(0, 8).flatMap(([name]) => [name, name.replace(/R$/, 'L')]),
  ]);
  gltf.scene.traverse(node => {
    if (node.isBone && wordJointNames.has(node.name)) {
      wordBones.push({ node, base: node.quaternion.clone() });
    }
  });
  for (const name of wordJointNames) {
    if (!wordBones.some(({ node }) => node.name === name)) {
      throw new Error(`The avatar model is missing the "${name}" signing joint.`);
    }
  }
  const wordBoneByName = new Map(wordBones.map(bone => [bone.node.name, bone]));
  const signingArmPose = new Map();
  let wristPoseApplied = false;
  for (const { node, base } of controlledBones) {
    node.quaternion.copy(base);
    const rotation = SIGNING_ARM_ROTATIONS[node.name];
    if (rotation) node.quaternion.set(...rotation).normalize();
  }
  gltf.scene.updateMatrixWorld(true);
  const upperArm = gltf.scene.getObjectByName('UpperArmR');
  const lowerArm = controlledBones.find(({ node }) => node.name === 'LowerArmR').node;
  const indexTip = gltf.scene.getObjectByName('Index2R_end');
  if (!indexTip) throw new Error('The avatar model is missing the "Index2R_end" articulation joint.');
  const upperArmParentRotation = upperArm.parent.getWorldQuaternion(new THREE.Quaternion());
  const armLift = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -0.55);
  upperArm.quaternion.premultiply(
    upperArmParentRotation.clone().invert().multiply(armLift).multiply(upperArmParentRotation),
  );
  gltf.scene.updateMatrixWorld(true);
  const handDirection = indexTip.getWorldPosition(new THREE.Vector3())
    .sub(lowerArm.getWorldPosition(new THREE.Vector3()))
    .normalize();
  const forwardDirection = new THREE.Vector3(0, 0, 1);
  const handSwing = new THREE.Quaternion().setFromUnitVectors(handDirection, forwardDirection);
  const lowerArmParentRotation = lowerArm.parent.getWorldQuaternion(new THREE.Quaternion());
  lowerArm.quaternion.premultiply(
    lowerArmParentRotation.clone().invert().multiply(handSwing).multiply(lowerArmParentRotation),
  );
  gltf.scene.updateMatrixWorld(true);
  const wristForward = indexTip.getWorldPosition(new THREE.Vector3())
    .sub(gltf.scene.getObjectByName('Palm1R').getWorldPosition(new THREE.Vector3()))
    .normalize();
  const wristAxis = wristForward.clone().cross(new THREE.Vector3(0, 1, 0)).normalize();
  const wristTiltWorld = new THREE.Quaternion().setFromAxisAngle(wristAxis, WRIST_TILT_RADIANS);
  const lowerArmWorldRotation = lowerArm.getWorldQuaternion(new THREE.Quaternion());
  const wristTilt = lowerArmWorldRotation.clone().invert()
    .multiply(wristTiltWorld)
    .multiply(lowerArmWorldRotation);
  for (const { node } of controlledBones) {
    const rotation = SIGNING_ARM_ROTATIONS[node.name];
    if (rotation) signingArmPose.set(node.name, node.quaternion.clone());
  }
  signingArmPose.set('UpperArmR', upperArm.quaternion.clone());
  signingArmPose.set('LowerArmR', lowerArm.quaternion.clone());
  for (const { node, base } of controlledBones) node.quaternion.copy(base);

  const applyWristPose = () => {
    if (!wristPoseApplied) return;
    for (const { node, base } of wristRoots) {
      node.quaternion.copy(wristTilt).multiply(base);
    }
    const thumb = controlledBones.find(({ node }) => node.name === 'ThumbR');
    thumb.node.quaternion.premultiply(wristTilt);
  };
  const restoreWristPose = () => {
    if (!wristPoseApplied) return;
    for (const { node, base } of wristRoots) node.quaternion.copy(base);
    const thumb = controlledBones.find(({ node }) => node.name === 'ThumbR');
    thumb.node.quaternion.premultiply(wristTilt.clone().invert());
    wristPoseApplied = false;
  };

  const letterPoses = new Map(Object.entries(LETTER_HANDSHAPES).map(([letter, shape]) => {
    const curl = level => [0, 0.52, 1.02][level];
    const angles = [
      [0, 0, curl(shape[0]) * 0.72],
      [0, 0, curl(shape[0]) * 0.58],
      [0, 0, curl(shape[1]) * 0.72],
      [0, 0, curl(shape[1]) * 0.58],
      [0, 0, curl(shape[2]) * 0.72],
      [0, 0, curl(shape[2]) * 0.58],
      [curl(shape[3]) * 0.34, -curl(shape[3]) * 0.22, shape[4] * 0.045],
      [curl(shape[3]) * 0.24, -curl(shape[3]) * 0.18, 0],
      null,
      null,
      null,
    ];
    return [letter, controlledBones.map(({ node, base }) => {
      const signingArmRotation = signingArmPose.get(node.name);
      if (signingArmRotation) return signingArmRotation.clone();
      const [x, y, z] = angles[jointIndex.get(node.name)];
      return base.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(x, y, z)));
    })];
  }));

  let fingerSpell = null;
  let wordSign = null;
  const makeWordPose = word => {
    const definition = SIGNKIT_WORD_POSES[word];
    if (!definition) throw new Error(`No adapted word sign is available for "${word}".`);
    const pose = new Map(wordBones.map(({ node, base }) => [node.name, base.clone()]));
    for (const [name, angles] of Object.entries(definition.joints)) {
      const bone = wordBoneByName.get(name);
      pose.set(name, bone.base.clone().multiply(
        new THREE.Quaternion().setFromEuler(new THREE.Euler(...angles)),
      ));
    }
    for (const side of ['L', 'R']) {
      const handshape = definition[`${side === 'L' ? 'left' : 'right'}Handshape`];
      if (!handshape) continue;
      const shape = LETTER_HANDSHAPES[handshape];
      const curl = level => [0, 0.52, 1.02][level];
      const angles = [
        [0, 0, curl(shape[0]) * 0.72],
        [0, 0, curl(shape[0]) * 0.58],
        [0, 0, curl(shape[1]) * 0.72],
        [0, 0, curl(shape[1]) * 0.58],
        [0, 0, curl(shape[2]) * 0.72],
        [0, 0, curl(shape[2]) * 0.58],
        [curl(shape[3]) * 0.34, -curl(shape[3]) * 0.22, shape[4] * 0.045],
        [curl(shape[3]) * 0.24, -curl(shape[3]) * 0.18, 0],
      ];
      FINGER_JOINTS.slice(0, 8).forEach(([rightName], index) => {
        const name = rightName.replace(/R$/, side);
        const bone = wordBoneByName.get(name);
        pose.set(name, bone.base.clone().multiply(
          new THREE.Quaternion().setFromEuler(new THREE.Euler(...angles[index])),
        ));
      });
    }
    return pose;
  };
  const applyWordPose = (from, pose, amount) => {
    for (const { node } of wordBones) {
      node.quaternion.copy(from.get(node.name)).slerp(pose.get(node.name), amount);
    }
  };
  const updateWordSign = delta => {
    if (!wordSign) return;
    wordSign.elapsed += delta;
    const transition = Math.min(0.22, wordSign.duration * 0.25);
    const holdEnd = wordSign.duration - transition;
    if (wordSign.elapsed <= transition) {
      const amount = wordSign.elapsed / transition;
      const eased = amount * amount * (3 - 2 * amount);
      applyWordPose(wordSign.from, wordSign.pose, eased);
    } else if (wordSign.elapsed <= holdEnd) {
      applyWordPose(wordSign.pose, wordSign.pose, 1);
    } else {
      const amount = Math.min(1, (wordSign.elapsed - holdEnd) / transition);
      const eased = amount * amount * (3 - 2 * amount);
      applyWordPose(wordSign.pose, wordSign.neutral, eased);
    }
    if (wordSign.elapsed >= wordSign.duration) {
      for (const { node, base } of wordBones) node.quaternion.copy(base);
      wordSign = null;
    }
  };
  const stopWordSign = () => {
    if (!wordSign) return;
    for (const { node, base } of wordBones) node.quaternion.copy(base);
    wordSign = null;
  };
  const applyPose = (from, to, amount) => {
    controlledBones.forEach(({ node }, index) => {
      node.quaternion.copy(from[index]).slerp(to[index], amount);
    });
    applyWristPose();
  };
  const capturePose = () => controlledBones.map(({ node }) => {
    const pose = node.quaternion.clone();
    if (wristPoseApplied && node.name === 'ThumbR') pose.premultiply(wristTilt.clone().invert());
    return pose;
  });
  const restoreNeutralPose = () => applyPose(capturePose(), neutralPose, 1);
  const updateFingerSpell = delta => {
    if (!fingerSpell) return;
    fingerSpell.elapsed += delta;
    const stepDuration = FINGER_SPELLING_LETTER_SECONDS / fingerSpell.speed;
    const moveDuration = Math.min(0.14 / fingerSpell.speed, stepDuration * 0.4);
    const target = letterPoses.get(fingerSpell.letters[fingerSpell.index]);
    const progress = Math.min(1, fingerSpell.elapsed / moveDuration);
    const eased = progress * progress * (3 - 2 * progress);

    if (fingerSpell.index === fingerSpell.letters.length - 1 &&
        fingerSpell.elapsed > stepDuration - moveDuration) {
      const returnProgress = Math.min(1, (fingerSpell.elapsed - (stepDuration - moveDuration)) / moveDuration);
      const returnEased = returnProgress * returnProgress * (3 - 2 * returnProgress);
      applyPose(target, neutralPose, returnEased);
    } else {
      applyPose(fingerSpell.from, target, eased);
    }

    if (fingerSpell.elapsed >= stepDuration) {
      fingerSpell.index += 1;
      if (fingerSpell.index === fingerSpell.letters.length) {
        restoreNeutralPose();
        fingerSpell = null;
        restoreWristPose();
      } else {
        fingerSpell.elapsed = 0;
        fingerSpell.from = capturePose();
      }
    }
  };

  let activeAction = idle;
  mixer.addEventListener('finished', event => {
    if (event.action !== activeAction || event.action === idle) return;
    activeAction.fadeOut(0.18);
    idle.reset().setEffectiveWeight(1).fadeIn(0.18).play();
    activeAction = idle;
  });

  return {
    model: gltf.scene,
    get fingerSpellingActive() {
      return fingerSpell !== null;
    },
    update(delta) {
      mixer.update(delta);
      for (const name of FINGER_BONES_TO_LENGTHEN) {
        const bone = controlledBones.find(({ node }) => node.name === name);
        if (bone) bone.node.scale.y = bone.baseScale.y * FINGER_LENGTH_SCALE;
      }
      updateFingerSpell(delta);
      updateWordSign(delta);
    },
    playFingerSpell(text, speed = 1) {
      const letters = [...String(text).toUpperCase()].filter(letter => letterPoses.has(letter));
      if (!letters.length) return 0;
      stopWordSign();
      const rate = Number.isFinite(speed) && speed > 0 ? speed : 1;
      fingerSpell = { letters, speed: rate, index: 0, elapsed: 0, from: capturePose() };
      wristPoseApplied = true;
      if (activeAction !== idle) activeAction.fadeOut(0.12);
      idle.reset().setEffectiveWeight(1).fadeIn(0.12).play();
      activeAction = idle;
      return letters.length * FINGER_SPELLING_LETTER_SECONDS / rate;
    },
    get wordSigningActive() {
      return wordSign !== null;
    },
    playWordSign(word, speed = 1) {
      const key = String(word).toUpperCase();
      const pose = makeWordPose(key);
      const rate = Number.isFinite(speed) && speed > 0 ? speed : 1;
      stopWordSign();
      wordSign = {
        elapsed: 0,
        duration: 1.15 / rate,
        pose,
        neutral: new Map(wordBones.map(({ node, base }) => [node.name, base.clone()])),
        from: new Map(wordBones.map(({ node }) => [node.name, node.quaternion.clone()])),
      };
      if (activeAction !== idle) activeAction.fadeOut(0.12);
      idle.reset().setEffectiveWeight(1).fadeIn(0.12).play();
      activeAction = idle;
      return wordSign.duration;
    },
    play(name, speed = 1) {
      const key = String(name).toLowerCase().replace(/[^a-z]/g, '');
      const action = actions.get(key);
      if (!action) throw new Error(`Avatar animation "${name}" is not available.`);
      stopWordSign();
      if (fingerSpell) {
        fingerSpell = null;
        restoreNeutralPose();
        restoreWristPose();
      }
      if (activeAction !== action) activeAction.fadeOut(0.18);
      action.reset();
      action.setLoop(THREE.LoopOnce, 1);
      action.clampWhenFinished = true;
      action.setEffectiveTimeScale(speed);
      action.setEffectiveWeight(1);
      action.fadeIn(0.18).play();
      activeAction = action;
      return action.getClip().duration / speed;
    },
  };
}
