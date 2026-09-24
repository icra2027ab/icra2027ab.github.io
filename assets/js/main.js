/* Project page interactivity. Plain JS, no build step. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(k));
    return n;
  };
  const fmt = (x, d = 1) => (x == null || isNaN(x) ? '·' : Number(x).toFixed(d));

  /* ---------------- theme ---------------- */
  const root = document.documentElement;
  $('#themeBtn').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
  });

  /* ---------------- math ---------------- */
  const renderMath = (node = document.body) => {
    if (window.renderMathInElement) {
      renderMathInElement(node, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'select', 'option'],
        throwOnError: false,
      });
    }
  };
  window.addEventListener('load', () => renderMath());

  /* ---------------- reveal + nav highlight + lazy autoplay ---------------- */
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add('vis'); io.unobserve(e.target); }
  }), { rootMargin: '0px 0px -8% 0px' });
  $$('.fade-in').forEach((n) => io.observe(n));

  const navLinks = $$('#navlinks a');
  const secObs = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    navLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
  }), { rootMargin: '-45% 0px -50% 0px' });
  ['video', 'method', 'sim', 'real', 'explorer', 'appendix'].forEach((id) => { const s = document.getElementById(id); if (s) secObs.observe(s); });

  const tocLinks = $$('#toc a');
  const tocObs = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    tocLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
  }), { rootMargin: '-20% 0px -70% 0px' });
  $$('.appx-sec').forEach((s) => tocObs.observe(s));

  const vidObs = new IntersectionObserver((es) => es.forEach((e) => {
    const v = e.target;
    if (e.isIntersecting) { if (v.preload === 'none') v.preload = 'auto'; v.play().catch(() => {}); } else v.pause();
  }), { threshold: 0.25 });
  $$('video.lazyplay').forEach((v) => { v.muted = true; v.removeAttribute('autoplay'); vidObs.observe(v); });

  /* ---------------- method stepper ---------------- */
  const STEPS = [
    { t: 'Retrieve', s: 'Most similar successful rollout', v: 'Fig1_Retrieve',
      d: '<b>Retrieve.</b> Given the OOD scene and instruction, pick the in-distribution rollout maximizing a weighted sum of DINOv2 image similarity and CLIP text similarity: $\\tau^* = \\arg\\max_i\\, \\beta\\,\\mathrm{sim}(\\psi_v(o\'_0), \\psi_v(\\tau_i[0])) + (1-\\beta)\\,\\mathrm{sim}(\\psi_\\ell(\\ell), \\psi_\\ell(\\ell_i))$.' },
    { t: 'Abstract', s: 'Perceive the demo, pick distance pairs', v: 'Fig1_Abstract',
      d: '<b>Abstract.</b> The retrieved rollout is segmented (Gemini 3 Flash boxes refined by SAM2) and split into phases (STRAP). For each phase a VLM picks entity pairs, a distance type (centroid, surface, orientation, gripper) and an optional axis or plane. Evaluating them per timestep gives reference trends $d^{n}_t$ that encode <em>how</em> the motion unfolds; the 3D view shows each pair and its projection axis or plane.' },
    { t: 'Perceive', s: 'Ground the live OOD scene', v: 'Fig1_Perception',
      d: '<b>Perceive.</b> At test time the same open-vocabulary pipeline runs on the live OOD scene: a box per task-relevant object, a SAM2 mask, and depth unprojection to metric centroids and surface points. Objects are matched to their counterparts in the demonstration (e.g. the new cylinder plays the role of the demo cube).' },
    { t: 'Align', s: 'Track progress, allow retries', v: 'Fig1_Alignment',
      d: '<b>Align.</b> The live scene is projected into the same distance space and each relation is aligned to the reference within a window around a shared cursor. The cursor can move backward, so the guidance follows retries instead of dragging the robot through unfinished stages.' },
    { t: 'Steer', s: 'Guide each denoising step', v: 'Fig1_SteeringLoop',
      d: '<b>Steer.</b> A differentiable dynamics model forecasts the distances a candidate chunk would induce; the gradient of the deviation from the reference, scaled by a golden-section line search with a do-no-harm guard, corrects the velocity field. Weights stay frozen.' },
  ];
  const stepsBox = $('#steps'), stepVideo = $('#stepVideo'), stepText = $('#stepText');
  let stepIdx = 0, stepTimer;
  const showStep = (i, user) => {
    stepIdx = i;
    $$('button', stepsBox).forEach((b, j) => b.classList.toggle('on', j === i));
    const s = STEPS[i];
    stepVideo.src = `assets/videos/method/${s.v}.mp4`;
    stepVideo.poster = `assets/videos/method/${s.v}.jpg`;
    stepVideo.play().catch(() => {});
    stepText.innerHTML = s.d;
    renderMath(stepText);
    if (user) clearInterval(stepTimer);
  };
  STEPS.forEach((s, i) => stepsBox.append(el('button', { onclick: () => showStep(i, true) },
    el('span', { class: 'n' }, `0${i + 1}`), el('b', {}, s.t), s.s)));
  stepVideo.loop = false;
  stepVideo.addEventListener('ended', () => showStep((stepIdx + 1) % STEPS.length));
  showStep(0);

  /* ---------------- bar charts ---------------- */
  const drawBars = (box, rows, { max = 100, ref = null, unit = '', dec = 1 } = {}) => {
    box.innerHTML = '';
    rows.forEach((r) => {
      const track = el('div', { class: 'bar-track' });
      const fills = r.stack || [{ v: r.v, cls: r.cls }];
      let acc = 0;
      fills.forEach((f) => {
        const fill = el('div', { class: `bar-fill ${f.cls || ''}`, title: f.title || '' });
        fill.style.position = 'absolute'; fill.style.left = (acc / max * 100) + '%';
        fill.style.borderRadius = '0';
        track.append(fill);
        requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = Math.max(0.4, f.v / max * 100) + '%'; }));
        acc += f.v;
      });
      if (ref != null) { const m = el('div', { class: 'bar-ref' }); m.style.left = (ref / max * 100) + '%'; m.style.borderLeft = '2px dashed var(--ink)'; m.style.background = 'none'; track.append(m); }
      const total = r.stack ? r.stack.reduce((a, b) => a + b.v, 0) : r.v;
      box.append(el('div', { class: 'bar-row' }, el('div', { class: 'lbl', html: r.label }), track,
        el('div', { class: 'val' }, r.valText || (fmt(total, dec) + unit))));
    });
  };

  const Q1 = {
    task: [26.3, 32.0, 17.3, 40.8], pos: [26.3, 27.6, 21.5, 34.3], avg: [26.3, 29.8, 19.4, 37.5],
    object: [83.0, 45.8, 18.3, 65.8], lang: [92.3, 61.5, 26.8, 75.0],
  };
  const q1Labels = ['No steering', 'VLS (full)', 'VLS (grad-only)', '<b>Ours</b>'];
  const q1Cls = ['base', 'vls', 'vls', 'ours'];
  const drawQ1 = (k) => drawBars($('#q1Bars'), Q1[k].map((v, i) => ({ label: q1Labels[i], v, cls: q1Cls[i] })), { ref: Q1[k][0] });
  const XP = { avg: [[26.3, 37.5], [9.9, 11.8]], task: [[26.3, 40.8], [19.3, 20.0]], pos: [[26.3, 34.3], [0.5, 3.5]] };
  const drawXP = (k) => drawBars($('#xpBars'), [
    { label: 'π<sub>0.5</sub> base', v: XP[k][0][0], cls: 'base' }, { label: '<b>π<sub>0.5</sub> + ours</b>', v: XP[k][0][1], cls: 'ours' },
    { label: 'GR00T N1.7 base', v: XP[k][1][0], cls: 'base' }, { label: '<b>GR00T N1.7 + ours</b>', v: XP[k][1][1], cls: 'ours' },
  ], { max: 60 });
  const bindSeg = (id, fn) => {
    const seg = $(id);
    seg.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      $$('button', seg).forEach((x) => x.classList.toggle('on', x === b)); fn(b.dataset.k);
    });
    fn($('button.on', seg).dataset.k);
  };
  bindSeg('#q1Seg', drawQ1);
  bindSeg('#xpSeg', drawXP);

  drawBars($('#profBars'), [
    { label: 'Base π<sub>0.5</sub>', stack: [{ v: 61.2, cls: 'base' }], valText: '61 · 1.0×' },
    { label: '<b>Ours</b>', stack: [{ v: 61.2, cls: 'base' }, { v: 11.8, cls: 'ours', title: 'guidance' }], valText: '73 · 1.2×' },
    { label: 'VLS (grad-only)', stack: [{ v: 61.2, cls: 'base' }, { v: 8.4, cls: 'vls' }], valText: '70 · 1.1×' },
    { label: 'VLS (full)', stack: [{ v: 61.2, cls: 'base' }, { v: 163.8, cls: 'vls' }, { v: 1005, cls: 'bad', title: 'particle resampling' }], valText: '1230 · 20×' },
  ], { max: 1230 });

  drawBars($('#ablBars'), [
    { label: 'No guidance', v: 26.3, cls: 'base' },
    { label: '+ gradient guidance (fixed step)', v: 23.8, cls: 'bad' },
    { label: '+ line search', v: 35.0, cls: 'ours' },
    { label: '<b>+ error normalization (ours)</b>', v: 37.5, cls: 'ours' },
  ], { max: 50, ref: 26.3 });

  /* ---------------- prompts ---------------- */
  $$('details.prompt').forEach((d) => {
    const load = () => {
      if (d.dataset.loaded) return; d.dataset.loaded = 1;
      fetch(d.dataset.src).then((r) => r.text()).then((t) => {
        $('pre', d).textContent = t;
        const sz = $('.sz', d); sz.textContent = `${t.split('\n').length} lines`;
        const btn = el('button', { class: 'copy', onclick: (e) => { e.preventDefault(); navigator.clipboard.writeText(t).then(() => { btn.textContent = 'Copied'; setTimeout(() => (btn.textContent = 'Copy'), 1200); }); } }, 'Copy');
        sz.after(btn);
      }).catch(() => { $('pre', d).textContent = 'Could not load this transcript.'; });
    };
    if (d.open) load();
    d.addEventListener('toggle', load);
  });

  /* ---------------- lightbox ---------------- */
  const lb = $('#lb'), lbVideo = $('#lbVideo'), lbInfo = $('#lbInfo');
  let lbList = [], lbIdx = 0;
  const infoFor = (it) => {
    const kv = [['Method', it.methodLabel], ['Outcome', it.success ? 'success' : (it.outcomeLabel || 'failure')]];
    (it.info || []).forEach((p) => kv.push(p));
    return kv.filter((p) => p[1] != null && p[1] !== '').map(([k, v]) => `<span>${k}: <b>${v}</b></span>`).join('');
  };
  const openLB = (list, i) => {
    lbList = list; lbIdx = i;
    const it = list[i];
    lbVideo.src = it.video; lbVideo.poster = it.poster || '';
    lbVideo.play().catch(() => {});
    lbInfo.innerHTML = infoFor(it);
    $('#lbPrev').disabled = i <= 0; $('#lbNext').disabled = i >= list.length - 1;
    lb.classList.add('open'); document.body.style.overflow = 'hidden';
  };
  const closeLB = () => { lb.classList.remove('open'); lbVideo.pause(); lbVideo.removeAttribute('src'); lbVideo.load(); document.body.style.overflow = ''; };
  $('#lbClose').onclick = closeLB;
  $('#lbPrev').onclick = () => lbIdx > 0 && openLB(lbList, lbIdx - 1);
  $('#lbNext').onclick = () => lbIdx < lbList.length - 1 && openLB(lbList, lbIdx + 1);
  lb.addEventListener('click', (e) => { if (e.target === lb) closeLB(); });
  document.addEventListener('keydown', (e) => {
    if (!lb.classList.contains('open')) return;
    if (e.key === 'Escape') closeLB();
    if (e.key === 'ArrowLeft') $('#lbPrev').click();
    if (e.key === 'ArrowRight') $('#lbNext').click();
  });

  /* ---------------- video tile ---------------- */
  const METHOD_LABEL = { base: 'Base', ours: 'Ours', ours_realperc: 'Ours · open-vocab perception', vls_full: 'VLS (full)', vls_grad: 'VLS (grad-only)' };
  const BADGE = { ours_realperc: ['realperc', 'ours · OV'], vls_full: ['vls', 'VLS'], vls_grad: ['vls', 'VLS grad'] };
  const tile = (it, list, i, { wide = false, caption = true } = {}) => {
    const t = el('div', { class: 'tile' + (wide ? ' wide' : ''), tabindex: 0, role: 'button', 'aria-label': `${it.methodLabel} rollout, ${it.success ? 'success' : 'failure'}: ${it.title}` });
    const img = el('img', { src: it.poster || '', alt: '', loading: 'lazy', decoding: 'async' });
    t.append(img,
      el('span', { class: 'badge ' + (BADGE[it.method] ? BADGE[it.method][0] : it.method) }, BADGE[it.method] ? BADGE[it.method][1] : it.method),
      el('span', { class: 'res ' + (it.success ? 's' : 'f'), title: it.success ? 'success' : 'failure' }, it.success ? '✓' : '✕'));
    if (caption) t.append(el('div', { class: 'meta' }, el('b', {}, it.title), it.sub || ''));
    let v;
    t.addEventListener('mouseenter', () => {
      if (!v) { v = el('video', { muted: true, loop: true, playsinline: true, preload: 'auto' }); v.muted = true; v.src = it.video; t.append(v); }
      v.play().then(() => t.classList.add('playing')).catch(() => {});
    });
    t.addEventListener('mouseleave', () => { if (v) { v.pause(); t.classList.remove('playing'); } });
    const open = () => openLB(list, i);
    t.addEventListener('click', open);
    t.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    return t;
  };

  /* ---------------- data ---------------- */
  const getJSON = (p) => fetch(p).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const PERT_LABEL = { task: 'Task', position: 'Position', swap: 'Position', object: 'Object', language: 'Language' };
  const SUITE_LABEL = { libero_10: 'LIBERO-10', libero_goal: 'LIBERO-Goal', libero_object: 'LIBERO-Object', libero_spatial: 'LIBERO-Spatial', positional: 'UR5 positional sweep', task: 'UR5 task variation' };
  const POLICY_LABEL = { pi05: 'π0.5', groot: 'GR00T N1.7' };
  const OUTCOME_LABEL = { grasp_miss: 'failure (grasp miss)', wrong_target: 'failure (wrong target)', dropped: 'failure (dropped)', collision: 'failure (collision)', other: 'failure (other)' };
  const normPert = (p) => (p === 'swap' ? 'position' : p);
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

  Promise.all([getJSON('assets/data/real_rollouts.json'), getJSON('assets/data/sim_rollouts.json'), getJSON('assets/data/sim_rates.json')])
    .then(([real, sim, rates]) => {
      real = Array.isArray(real) ? real : (real && real.rollouts) || [];
      sim = Array.isArray(sim) ? sim : (sim && sim.rollouts) || [];

      const realItems = real.map((r) => ({
        raw: r, domain: 'real', policy: 'pi05', method: r.method, methodLabel: METHOD_LABEL[r.method] || r.method,
        group: r.experiment, pert: r.experiment === 'positional' ? 'position' : 'task', success: !!r.success,
        video: r.video, poster: r.poster, outcomeLabel: OUTCOME_LABEL[r.outcome],
        title: r.experiment === 'positional' ? `UR5 · cell ${r.cell} · trial ${r.trial ?? ''}` : `UR5 · "${r.target}" · layout ${r.layout || r.trial}`,
        sub: r.experiment === 'positional' ? `${fmt(r.sigma)}σ · ${r.zone || ''}` : (r.reached ? `reached ${r.reached}` : ''),
        info: r.experiment === 'positional'
          ? [['Experiment', 'positional sweep'], ['Cell', r.cell], ['Distance', r.sigma != null ? fmt(r.sigma, 2) + 'σ' : null], ['Zone', r.zone], ['Trial', r.trial], ['Clip', r.clip_kind === 'dashboard_slowed' ? 'one frame per inference, slowed 3×' : 'real time']]
          : [['Experiment', 'task variation'], ['Prompted', r.target], ['Reached', r.reached], ['Layout', r.layout], ['Clip', r.clip_kind === 'dashboard_slowed' ? 'one frame per inference, slowed 3×' : 'real time']],
      }));
      const simItems = sim.map((r) => ({
        raw: r, domain: 'sim', policy: r.policy, method: r.method, methodLabel: METHOD_LABEL[r.method] || r.method,
        group: r.suite, pert: normPert(r.perturbation), success: !!r.success, video: r.video, poster: r.poster,
        title: r.instruction || `task ${r.task_id}`,
        sub: `${POLICY_LABEL[r.policy] || r.policy} · ${SUITE_LABEL[r.suite] || r.suite} · ${PERT_LABEL[r.perturbation] || r.perturbation}`,
        info: [['Policy', POLICY_LABEL[r.policy] || r.policy], ['Suite', SUITE_LABEL[r.suite] || r.suite], ['Shift', PERT_LABEL[r.perturbation] || r.perturbation],
          ['Task', r.instruction], ['Episode', r.episode], r.dashboard ? ['Dashboard', `<a href="${r.dashboard}" target="_blank" style="color:#9cc2ff">with guidance plots ↗</a>`] : null].filter(Boolean),
      }));
      const all = [...simItems, ...realItems];
      $('#statVideos').textContent = all.length ? all.length.toLocaleString() : '·';

      initReal(realItems);
      initRates(rates, simItems);
      initPairs(simItems);
      initExplorer(all);
    });

  /* ---------------- real robot: positional grid ---------------- */
  function initReal(items) {
    const pos = items.filter((i) => i.group === 'positional');
    const task = items.filter((i) => i.group === 'task');
    const rows = ['A', 'B', 'C', 'D'], cols = [1, 2, 3, 4];
    const ZONE = { A: 'near', B: 'in', C: 'near', D: 'far' };
    const zoneOf = (cell) => (cell === 'B4' ? 'near' : ZONE[cell[0]]);
    let selCell = null;

    const cellView = $('#cellView');
    const showCell = (cell) => {
      selCell = cell;
      $$('.cellgrid .cell').forEach((c) => c.classList.toggle('sel', c.dataset.cell === cell));
      const b = pos.filter((i) => i.raw.cell === cell && i.method === 'base');
      const o = pos.filter((i) => i.raw.cell === cell && i.method === 'ours');
      const list = [...b, ...o];
      const sig = list.map((i) => i.raw.sigma).filter((x) => x != null);
      cellView.innerHTML = '';
      cellView.append(el('div', { class: 'hdr' },
        el('b', {}, `Cell ${cell}`),
        el('span', { class: 'muted small' }, `${zoneOf(cell)}-${zoneOf(cell) === 'in' ? 'distribution' : 'OOD'}${sig.length ? ` · measured ${fmt(Math.min(...sig), 1)} to ${fmt(Math.max(...sig), 1)}σ` : ''} · base ${b.filter((i) => i.success).length}/${b.length}, ours ${o.filter((i) => i.success).length}/${o.length}`)));
      const col = (title, cls, arr) => el('div', {}, el('h5', { class: cls }, title),
        arr.length ? el('div', { class: 'tiles' }, arr.map((it) => tile(it, list, list.indexOf(it), { wide: true }))) : el('div', { class: 'empty' }, 'No video for this cell.'));
      cellView.append(el('div', { class: 'cmp' }, col('Base (no steering)', 'base', b), col('Ours', 'ours', o)));
    };

    const drawGrid = (box, method, sumBox) => {
      box.innerHTML = '';
      box.append(el('div', { class: 'hd' }));
      cols.forEach((c) => box.append(el('div', { class: 'hd' }, String(c))));
      rows.forEach((r) => {
        box.append(el('div', { class: 'hd' }, r));
        cols.forEach((c) => {
          const cell = r + c;
          const its = pos.filter((i) => i.method === method && i.raw.cell === cell).sort((a, b) => (a.raw.trial ?? 0) - (b.raw.trial ?? 0));
          const k = its.filter((i) => i.success).length, n = its.length;
          const z = zoneOf(cell);
          const node = el('div', { class: `cell z-${z}`, 'data-cell': cell, title: `${cell}: ${k}/${n} success`, onclick: () => showCell(cell) },
            el('div', { class: 'dots' }, its.map((i) => el('i', { class: i.success ? 's' : 'f' }))),
            el('div', { class: 'nm' }, n ? `${k}/${n}` : cell));
          const frac = n ? k / n : 0;
          node.style.background = `color-mix(in srgb, var(--green) ${Math.round(frac * 38)}%, var(--surface-2))`;
          box.append(node);
        });
      });
      const all = pos.filter((i) => i.method === method);
      sumBox.textContent = all.length ? `${all.filter((i) => i.success).length}/${all.length} success` : '';
    };
    if (pos.length) {
      drawGrid($('#gridBase'), 'base', $('#gBaseSum'));
      drawGrid($('#gridOurs'), 'ours', $('#gOursSum'));
      showCell('A2');
    } else {
      $('#gridBase').parentElement.parentElement.innerHTML = '<div class="empty" style="grid-column:1/-1">Real-robot rollouts are loading or unavailable.</div>';
    }
    const kp = (a, b, lab) => el('div', { class: 'kpi' }, el('b', { html: `${a} → <span style="color:var(--accent)">${b}</span>` }), el('span', {}, lab));
    $('#posKpis').append(kp('29.6', '63.0%', 'near-OOD success (2 to 4.5σ), p = 0.028'), kp('35.4', '54.2%', 'full sweep, 48 placements per method'), kp('2.8σ', '3.4σ', 'logistic 50% crossing'));

    /* task variation confusion */
    const OBJ = ['cable', 'banana', 'spoon'];
    const confView = $('#confView');
    const drawConf = (box, method, sumBox) => {
      const its = task.filter((i) => i.method === method);
      box.innerHTML = '';
      box.append(el('div', { class: 'h' }, 'prompt ↓ / reached →'));
      [...OBJ, 'none'].forEach((o) => box.append(el('div', { class: 'h' }, o)));
      OBJ.forEach((tgt) => {
        box.append(el('div', { class: 'h r' }, tgt));
        const rowIts = its.filter((i) => i.raw.target === tgt);
        [...OBJ, 'none'].forEach((rch) => {
          const m = rowIts.filter((i) => (OBJ.includes(i.raw.reached) ? i.raw.reached : 'none') === rch);
          const frac = rowIts.length ? m.length / rowIts.length : 0;
          const good = rch === tgt;
          const c = el('div', { class: 'c' + (m.length ? '' : ' zero'), title: `${m.length}/${rowIts.length}` }, m.length ? `${m.length}` : '0');
          c.style.background = m.length ? `color-mix(in srgb, ${good ? 'var(--green)' : 'var(--red)'} ${Math.round(12 + frac * 55)}%, var(--surface))` : 'var(--surface-2)';
          if (m.length) c.addEventListener('click', () => {
            $$('.conf .c').forEach((x) => x.classList.remove('sel')); c.classList.add('sel');
            confView.innerHTML = '';
            confView.append(el('div', { class: 'hdr' }, el('b', {}, `${method === 'ours' ? 'Ours' : 'Base'}: prompted "${tgt}", reached ${rch}`), el('span', { class: 'muted small' }, `${m.length} trial${m.length > 1 ? 's' : ''}`)),
              el('div', { class: 'tiles' }, m.map((it, j) => tile(it, m, j, { wide: true }))));
          });
          box.append(c);
        });
      });
      const conf = its.filter((i) => i.raw.reached && i.raw.reached !== i.raw.target && OBJ.includes(i.raw.reached)).length;
      sumBox.textContent = its.length ? `${its.filter((i) => i.success).length}/${its.length} bagged · ${conf} confused` : '';
    };
    if (task.length) {
      drawConf($('#confBase'), 'base', $('#cBaseSum'));
      drawConf($('#confOurs'), 'ours', $('#cOursSum'));
      const b = task.filter((i) => i.method === 'base'), o = task.filter((i) => i.method === 'ours');
      confView.innerHTML = '';
      confView.append(el('div', { class: 'cmp' },
        el('div', {}, el('h5', { class: 'base' }, 'Base: all trials'), el('div', { class: 'tiles' }, b.map((it, j) => tile(it, b, j, { wide: true })))),
        el('div', {}, el('h5', { class: 'ours' }, 'Ours: all trials'), el('div', { class: 'tiles' }, o.map((it, j) => tile(it, o, j, { wide: true }))))));
    } else {
      $('#confBase').parentElement.parentElement.innerHTML = '<div class="empty" style="grid-column:1/-1">Real-robot rollouts are loading or unavailable.</div>';
    }
  }

  /* ---------------- sim: rates heat table ---------------- */
  const SUITES = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10'];
  const FOCUS_S = ['libero_object', 'libero_spatial'], FOCUS_P = ['task', 'position'];
  const isFocus = (i) => i.domain === 'real' || (FOCUS_S.includes(i.group) && FOCUS_P.includes(i.pert));
  const PERTS = ['task', 'position', 'object', 'language'];
  const getRate = (rates, policy, method, suite, pert) => {
    const m = rates?.[policy]?.[method]?.[suite];
    if (!m) return null;
    const v = m[pert] ?? (pert === 'position' ? m.swap ?? m.pos : null);
    if (v == null) return null;
    const r = typeof v === 'object' ? v.rate : v;
    return { rate: r, n: typeof v === 'object' ? v.n : null, reported: !!(v && v.reported) };
  };
  function initRates(rates, simItems) {
    const seg = $('#rateSeg');
    const policies = rates ? Object.keys(rates) : [];
    if (!policies.length) { $('#rateTable').innerHTML = '<tbody><tr><td class="muted">Per-suite rates unavailable.</td></tr></tbody>'; return; }
    policies.forEach((p, i) => seg.append(el('button', { 'data-k': p, class: i === 0 ? 'on' : '' }, POLICY_LABEL[p] || p)));
    const draw = (policy) => {
      const tb = el('tbody');
      tb.append(el('tr', {}, el('th', {}, ''), PERTS.map((p) => el('th', {}, PERT_LABEL[p]))));
      SUITES.forEach((s) => {
        const tr = el('tr', {}, el('td', { class: 'rowh' }, SUITE_LABEL[s]));
        PERTS.forEach((p) => {
          const b = getRate(rates, policy, 'base', s, p), o = getRate(rates, policy, 'ours', s, p);
          const focus = FOCUS_S.includes(s) && FOCUS_P.includes(p);
          const td = el('td', { class: focus ? 'focus' : 'dim' });
          if (!b && !o) { td.textContent = '·'; td.style.background = 'var(--surface-2)'; td.style.cursor = 'default'; tr.append(td); return; }
          const d = b && o ? o.rate - b.rate : null;
          td.innerHTML = `<b>${o ? fmt(o.rate, 0) : '·'}</b><span class="d">base ${b ? fmt(b.rate, 0) + (b.reported ? '*' : '') : '·'}${d != null ? ` · ${d >= 0 ? '+' : ''}${fmt(d, 0)}` : ''}</span>`;
          const a = d == null ? 0 : Math.min(60, Math.abs(d) * 1.6);
          td.style.background = `color-mix(in srgb, ${d != null && d < 0 ? 'var(--red)' : 'var(--green)'} ${Math.round(8 + a)}%, var(--surface))`;
          td.title = `${SUITE_LABEL[s]} / ${PERT_LABEL[p]}: ours ${o ? fmt(o.rate) : '·'}% (n=${o?.n ?? '?'}), base ${b ? fmt(b.rate) : '·'}% (${b?.reported ? 'reported in paper' : 'n=' + (b?.n ?? '?')})`;
          td.addEventListener('click', () => {
            $$('#rateTable td').forEach((x) => x.classList.remove('sel')); td.classList.add('sel');
            setPairs(policy, s, p);
            document.getElementById('simPairsTitle').scrollIntoView({ behavior: 'smooth', block: 'start' });
          });
          tr.append(td);
        });
        tb.append(tr);
      });
      const t = $('#rateTable'); t.innerHTML = ''; t.append(tb);
      $('#rateCap').innerHTML = `<b>Ours</b> (bold) with the unsteered base and the difference in points, ${POLICY_LABEL[policy] || policy}, GT labels, 100 episodes per cell. Green: guidance helps; red: guidance hurts.`;
    };
    seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; $$('button', seg).forEach((x) => x.classList.toggle('on', x === b)); draw(b.dataset.k); });
    draw(policies[0]);

    // full per-suite table in the appendix
    const wrap = $('#rateFullWrap');
    const methods = ['base', 'ours', 'ours_realperc', 'vls_full', 'vls_grad'];
    const tbl = el('table', { class: 't' });
    tbl.append(el('thead', {}, el('tr', {}, el('th', {}, 'Policy / method'), el('th', {}, 'Suite'), PERTS.map((p) => el('th', {}, PERT_LABEL[p])))));
    const tb = el('tbody');
    policies.forEach((pol) => methods.forEach((m) => {
      if (!rates[pol][m]) return;
      SUITES.forEach((s, si) => {
        const cells = PERTS.map((p) => { const r = getRate(rates, pol, m, s, p); return el('td', {}, r ? `${fmt(r.rate)}${r.reported ? '*' : r.n ? ` (${r.n})` : ''}` : '·'); });
        tb.append(el('tr', { class: m === 'ours' ? 'ours' : '' }, el('td', {}, si === 0 ? `${POLICY_LABEL[pol] || pol} · ${METHOD_LABEL[m]}` : ''), el('td', {}, SUITE_LABEL[s]), cells));
      });
    }));
    tbl.append(tb);
    wrap.append(el('h4', {}, 'Per-suite success rates of the runs hosted on this page'), el('div', { class: 'tbl-wrap' }, tbl),
      el('p', { class: 'tcap' }, 'Success rate (%) with the number of episodes in parentheses, computed from the run summaries. Suite averages reproduce the tables above.'));
  }

  /* ---------------- sim: base vs ours pairs ---------------- */
  let setPairs = () => {};
  function initPairs(simItems) {
    const box = $('#pairs');
    if (!simItems.length) { box.innerHTML = '<div class="empty">Simulation rollouts are loading or unavailable.</div>'; return; }
    const sSel = $('#pairSuite'), pSel = $('#pairPert'), polSel = $('#pairPolicy');
    const uniq = (a) => [...new Set(a)];
    uniq(simItems.map((i) => i.group)).sort((a, b) => SUITES.indexOf(a) - SUITES.indexOf(b)).forEach((s) => sSel.append(el('option', { value: s }, SUITE_LABEL[s] || s)));
    PERTS.filter((p) => simItems.some((i) => i.pert === p)).forEach((p) => pSel.append(el('option', { value: p }, PERT_LABEL[p])));
    uniq(simItems.map((i) => i.policy)).forEach((p) => polSel.append(el('option', { value: p }, POLICY_LABEL[p] || p)));
    const draw = () => {
      const pol = polSel.value, s = sSel.value, p = pSel.value;
      const its = simItems.filter((i) => i.policy === pol && i.group === s && i.pert === p);
      const base = its.filter((i) => i.method === 'base'), ours = its.filter((i) => i.method === 'ours');
      box.innerHTML = '';
      const pairs = [];
      const used = new Set();
      ours.forEach((o) => {
        const b = base.find((x) => !used.has(x) && x.raw.task_id === o.raw.task_id && x.raw.episode === o.raw.episode)
          || base.find((x) => !used.has(x) && x.raw.task_id === o.raw.task_id);
        if (b) used.add(b);
        pairs.push([b, o]);
      });
      base.filter((b) => !used.has(b)).forEach((b) => pairs.push([b, null]));
      const r = (m) => `${its.filter((i) => i.method === m && i.success).length}/${its.filter((i) => i.method === m).length}`;
      const vls = its.filter((i) => i.method === 'vls_full' || i.method === 'vls_grad');
      const addVls = () => { if (vls.length) box.append(el('p', { class: 'small muted', style: 'grid-column:1/-1;margin:8px 0 0' }, `VLS baseline in this cell: full ${r('vls_full')}, grad-only ${r('vls_grad')} success in this sample (initial state 0 of each task).`), el('div', { class: 'tiles', style: 'grid-column:1/-1' }, vls.map((it, j) => tile(it, vls, j)))); };
      if (!base.length && its.length) {
        const g = [...ours, ...its.filter((i) => i.method === 'ours_realperc')];
        box.append(el('p', { class: 'small muted', style: 'grid-column:1/-1;margin:0' }, `No unsteered footage was retained for this cell, so only guided rollouts are shown (${ours.filter((i) => i.success).length}/${ours.length} success in this sample).`),
          el('div', { class: 'tiles', style: 'grid-column:1/-1' }, g.map((it, j) => tile(it, g, j))));
        addVls();
        return;
      }
      if (!pairs.length) { box.innerHTML = '<div class="empty">No hosted rollouts for this combination (e.g. no unsteered footage exists for it).</div>'; return; }
      pairs.sort((x, y) => (!!(y[0] && y[1]) - !!(x[0] && x[1]))).slice(0, 18).forEach(([b, o]) => {
        const list = [b, o].filter(Boolean);
        const same = b && o && b.raw.episode === o.raw.episode;
        box.append(el('div', { class: 'pair' },
          el('div', { class: 'two' },
            b ? tile(b, list, 0, { caption: false }) : el('div', { class: 'tile empty small', style: 'display:grid;place-items:center;padding:8px' }, 'no base clip'),
            o ? tile(o, list, list.indexOf(o), { caption: false }) : el('div', { class: 'tile empty small', style: 'display:grid;place-items:center;padding:8px' }, 'no guided clip')),
          el('div', { class: 'inst' }, (o || b).title, ' ', el('span', {}, `· task ${(o || b).raw.task_id ?? ''}${same ? ` · ep ${o.raw.episode} (same init)` : ''}`))));
      });
      addVls();
      box.prepend(el('p', { class: 'small muted', style: 'grid-column:1/-1;margin:0' }, `Hosted sample for this cell: base ${r('base')} success, ours ${r('ours')} success. Left tile: base; right tile: ours. Hover to play, click to enlarge.`));
    };
    [sSel, pSel, polSel].forEach((x) => x.addEventListener('change', draw));
    setPairs = (pol, s, p) => { polSel.value = pol; sSel.value = s; pSel.value = p; draw(); };
    if ([...pSel.options].some((o) => o.value === 'position')) pSel.value = 'position';
    sSel.value = 'libero_object'; pSel.value = 'task';
    $('#focusSeg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; $$('#focusSeg button').forEach((x) => x.classList.toggle('on', x === b)); setPairs('pi05', b.dataset.s, b.dataset.p); document.getElementById('simPairsTitle').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
    draw();
  }

  /* ---------------- explorer ---------------- */
  function initExplorer(all) {
    const f = { domain: $('#fDomain'), method: $('#fMethod'), policy: $('#fPolicy'), suite: $('#fSuite'), pert: $('#fPert'), outcome: $('#fOutcome') };
    const uniq = (a) => [...new Set(a)].filter(Boolean);
    uniq(all.map((i) => i.policy)).forEach((p) => f.policy.append(el('option', { value: p }, POLICY_LABEL[p] || p)));
    uniq(all.map((i) => i.group)).forEach((s) => f.suite.append(el('option', { value: s }, SUITE_LABEL[s] || s)));
    uniq(all.map((i) => i.pert)).forEach((p) => f.pert.append(el('option', { value: p }, PERT_LABEL[p] || cap(p))));
    $('#exTotal').textContent = all.length.toLocaleString();
    const PAGE = 48;
    let shown = PAGE, cur = [];
    const grid = $('#exTiles'), more = $('#exMore');
    // interleave so the first page is varied
    const order = all.map((it, i) => ({ it, k: ((i * 2654435761) >>> 0) % 100000 })).sort((a, b) => (isFocus(b.it) - isFocus(a.it)) || (a.k - b.k)).map((x) => x.it);
    const apply = () => {
      cur = order.filter((i) => (!f.domain.value || i.domain === f.domain.value)
        && (!f.method.value || i.method === f.method.value)
        && (!f.policy.value || i.policy === f.policy.value)
        && (!f.suite.value || i.group === f.suite.value)
        && (!f.pert.value || i.pert === f.pert.value)
        && (!f.outcome.value || (f.outcome.value === 's') === i.success));
      shown = PAGE; render();
    };
    const render = () => {
      grid.innerHTML = '';
      $('#exCount').textContent = cur.length.toLocaleString();
      if (!cur.length) { grid.append(el('div', { class: 'empty', style: 'grid-column:1/-1' }, all.length ? 'No rollouts match these filters.' : 'Rollout videos are loading or unavailable.')); more.hidden = true; return; }
      cur.slice(0, shown).forEach((it, i) => grid.append(tile(it, cur, i)));
      more.hidden = shown >= cur.length;
      more.textContent = `Show more (${Math.min(PAGE, cur.length - shown)} of ${cur.length - shown} remaining)`;
    };
    more.addEventListener('click', () => { const start = shown; shown += PAGE; cur.slice(start, shown).forEach((it, i) => grid.append(tile(it, cur, start + i))); more.hidden = shown >= cur.length; more.textContent = `Show more (${Math.min(PAGE, cur.length - shown)} of ${cur.length - shown} remaining)`; });
    Object.values(f).forEach((s) => s.addEventListener('change', apply));
    apply();
  }
})();
