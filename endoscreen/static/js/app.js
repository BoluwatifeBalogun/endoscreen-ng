/* EndoScreen NG front-end behaviour. Vanilla JS, no dependencies.
   Every enhancement is optional: with JavaScript off the forms still post
   and the server validates everything again. */
(() => {
  'use strict';
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- Navigation ---- */
  const navWrap = $('[data-nav]');
  if (navWrap) {
    const onScroll = () => navWrap.classList.toggle('scrolled', window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    const toggle = $('[data-nav-toggle]'), links = $('#nav-links');
    toggle?.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open);
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
  }

  /* ---- Toasts ---- */
  $$('.toast').forEach((toast) => {
    const close = () => { toast.classList.add('out'); setTimeout(() => toast.remove(), 300); };
    $('[data-dismiss]', toast)?.addEventListener('click', close);
    if (!toast.classList.contains('toast-error')) setTimeout(close, 6000);
  });
  $('[data-autofocus]')?.focus();

  /* ---- Headline word reveal ---- */
  $$('[data-split]').forEach((el) => {
    const words = el.textContent.trim().split(/\s+/);
    el.setAttribute('aria-label', el.textContent.trim());
    el.innerHTML = words.map((w, i) =>
      `<span class="word" aria-hidden="true"><span style="--w:${i}">${w}</span></span>`).join(' ');
  });

  /* ---- Number ticker ---- */
  const tick = (el) => {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const suffix = el.dataset.suffix || '', prefix = el.dataset.prefix || '';
    if (reduced || !isFinite(target)) return;
    const start = performance.now(), duration = 1200;
    const frame = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };

  /* ---- Gauge draw ---- */
  const drawGauge = (arc) => {
    const value = Math.max(0, Math.min(1, parseFloat(arc.dataset.gauge)));
    if (reduced) { arc.style.strokeDashoffset = 100 * (1 - value); return; }
    arc.style.transition = 'none';
    arc.style.strokeDashoffset = 100;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      arc.style.transition = '';
      arc.style.strokeDashoffset = 100 * (1 - value);
    }));
  };

  /* ---- Scroll reveal (also triggers tickers and gauges once visible) ---- */
  const seen = new WeakSet();
  const activate = (el) => {
    if (seen.has(el)) return;
    seen.add(el);
    el.classList.add('in');
    const counters = el.matches('[data-count]') ? [el] : $$('[data-count]', el);
    counters.forEach(tick);
    (el.matches('[data-gauge]') ? [el] : $$('[data-gauge]', el)).forEach(drawGauge);
  };
  const targets = new Set([...$$('.reveal'), ...$$('[data-count]'), ...$$('[data-gauge]')]
    .filter((el) => el.matches('.reveal') || !el.closest('.reveal')));
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) { activate(e.target); io.unobserve(e.target); }
    }), { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });
    targets.forEach((el) => io.observe(el));
  } else {
    targets.forEach(activate);
  }

  /* ---- Spotlight cards ---- */
  $$('.spot').forEach((card) => card.addEventListener('pointermove', (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty('--mx', `${e.clientX - r.left}px`);
    card.style.setProperty('--my', `${e.clientY - r.top}px`);
  }));

  /* ---- Small utilities ---- */
  $$('[data-toggle-password]').forEach((btn) => btn.addEventListener('click', () => {
    const input = document.getElementById(btn.dataset.togglePassword);
    const show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    btn.setAttribute('aria-pressed', show);
    btn.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
  }));
  $$('[data-print]').forEach((btn) => btn.addEventListener('click', () => {
    $$('details').forEach((d) => { d.open = true; });
    window.print();
  }));
  $$('[data-autosubmit]').forEach((el) => el.addEventListener('change', () => el.form.submit()));

  /* ---- Confirm dialog for destructive actions ---- */
  const dialog = $('#confirm-dialog');
  $$('form[data-confirm]').forEach((form) => form.addEventListener('submit', (e) => {
    if (form.dataset.confirmed || !dialog?.showModal) return;
    e.preventDefault();
    $('#confirm-title').textContent = form.dataset.confirmTitle;
    $('#confirm-body').textContent = form.dataset.confirmBody;
    $('#confirm-ok').textContent = form.dataset.confirmLabel || 'Delete';
    dialog.returnValue = 'cancel';
    dialog.showModal();
    dialog.addEventListener('close', () => {
      if (dialog.returnValue === 'confirm') { form.dataset.confirmed = '1'; form.submit(); }
    }, { once: true });
  }));

  /* ---- Field validation shared by every form ---- */
  const messageFor = (input) => {
    const v = input.validity, label = input.dataset.label || 'This field';
    if (input.dataset.match && input.value !== document.getElementById(input.dataset.match).value) {
      return input.dataset.msg || 'Values do not match.';
    }
    if (v.valid) return '';
    if (input.dataset.msg && !v.rangeUnderflow && !v.rangeOverflow) return input.dataset.msg;
    if (v.valueMissing) return input.type === 'radio' ? `Choose an option for ${label.toLowerCase()}.` : `${label} is required.`;
    if (v.rangeUnderflow || v.rangeOverflow) {
      const unit = input.dataset.unitNow || input.dataset.unit || '';
      return `${label} must be between ${input.min} and ${input.max} ${unit}.`.replace(' .', '.');
    }
    if (v.badInput || v.typeMismatch) return `${label} is not in a valid format.`;
    if (v.stepMismatch) return '';
    return input.dataset.msg || `${label} is not valid.`;
  };
  const holderOf = (input) => input.closest('.field') || input.closest('.sym-group') || input.parentElement;
  const setError = (input, message) => {
    const holder = holderOf(input);
    const id = `${input.name}-error`;
    let node = holder.querySelector(`[id="${id}"]`);
    if (!message) {
      node?.remove();
      holder.classList.remove('has-error');
      input.removeAttribute('aria-invalid');
      return;
    }
    if (!node) {
      node = document.createElement('p');
      node.className = 'field-error';
      node.id = id;
      node.setAttribute('role', 'alert');
      holder.appendChild(node);
    }
    node.textContent = message;
    holder.classList.add('has-error');
    input.setAttribute('aria-invalid', 'true');
    const described = (input.getAttribute('aria-describedby') || '').split(' ').filter(Boolean);
    if (!described.includes(id)) input.setAttribute('aria-describedby', [...described, id].join(' '));
  };
  const validateInputs = (inputs) => {
    let first = null;
    const radiosDone = new Set();
    inputs.forEach((input) => {
      if (input.disabled || input.type === 'hidden') return;
      if (input.type === 'radio') {
        if (radiosDone.has(input.name)) return;
        radiosDone.add(input.name);
      }
      const message = messageFor(input);
      setError(input, message);
      if (message && !first) first = input;
    });
    return first;
  };
  /* Errors are added only on Continue/Submit and cleared as soon as the value
     becomes valid. Nothing is validated on blur: an error appearing or vanishing
     at that moment shifts the layout and makes the next click miss its button. */
  const liveClear = (form) => ['input', 'change'].forEach((type) => form.addEventListener(type, (e) => {
    const el = e.target;
    if (!el.matches('input, select')) return;
    const group = el.type === 'radio' ? $$(`input[name="${el.name}"]`, form) : [el];
    if (group.some((i) => i.getAttribute('aria-invalid')) && !messageFor(el)) group.forEach((i) => setError(i, ''));
    $$('[data-match]', form).forEach((m) => {
      if (m.getAttribute('aria-invalid') && !messageFor(m)) setError(m, '');
    });
  }));
  const startLoading = (form) => {
    const btn = form.querySelector('[type=submit][data-loading-text]');
    if (!btn) return;
    btn.classList.add('is-loading');
    btn.setAttribute('aria-busy', 'true');
    btn.textContent = btn.dataset.loadingText;
  };
  $$('form[data-validate]').forEach((form) => {
    const inputs = () => $$('input, select', form);
    form.addEventListener('submit', (e) => {
      const first = validateInputs(inputs());
      if (first) { e.preventDefault(); first.focus(); return; }
      startLoading(form);
    });
    liveClear(form);
  });
  $$('form:not([data-validate]):not([data-wizard]):not([data-confirm])').forEach((form) =>
    form.addEventListener('submit', () => { if (form.checkValidity()) startLoading(form); }));

  /* ---- Screening wizard ---- */
  const wizard = $('[data-wizard]');
  if (wizard) {
    const steps = $$('[data-step]', wizard);
    const dots = $$('[data-step-dot]', wizard);
    const bar = $('[data-step-bar]', wizard);
    const prev = $('[data-prev]', wizard), next = $('[data-next]', wizard), submit = $('[data-submit]', wizard);
    const count = $('[data-step-count]', wizard), status = $('[data-step-status]', wizard);
    let current = 0;

    const stepOf = (el) => steps.findIndex((s) => s.contains(el));
    const show = (index, focus = true) => {
      wizard.style.setProperty('--dir', index >= current ? 1 : -1);
      current = index;
      steps.forEach((s, i) => s.classList.toggle('active', i === index));
      dots.forEach((d, i) => {
        d.classList.toggle('done', i < index);
        d.classList.toggle('current', i === index);
        if (i === index) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
      });
      bar.style.width = `${((index + 1) / steps.length) * 100}%`;
      prev.hidden = index === 0;
      next.hidden = index === steps.length - 1;
      submit.hidden = index !== steps.length - 1;
      const title = steps[index].querySelector('legend').lastChild.textContent.trim();
      count.textContent = `Step ${index + 1} of ${steps.length}`;
      status.textContent = `Step ${index + 1} of ${steps.length}: ${title}`;
      if (focus) {
        const target = steps[index].querySelector('input:not([type=hidden]), select');
        window.scrollTo({ top: wizard.getBoundingClientRect().top + window.scrollY - 110, behavior: reduced ? 'auto' : 'smooth' });
        target?.focus({ preventScroll: true });
      }
    };
    const validateStep = (index) => validateInputs($$('input, select', steps[index]));

    next.addEventListener('click', () => {
      const first = validateStep(current);
      if (first) { first.focus(); return; }
      show(current + 1);
    });
    prev.addEventListener('click', () => show(current - 1));
    wizard.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.matches('input:not([type=checkbox]):not([type=radio])') && current < steps.length - 1) {
        e.preventDefault();
        next.click();
      }
    });
    wizard.addEventListener('submit', (e) => {
      for (let i = 0; i < steps.length; i += 1) {
        const first = validateStep(i);
        if (first) { e.preventDefault(); show(i, false); first.focus(); return; }
      }
      startLoading(wizard);
    });
    liveClear(wizard);

    /* live BMI */
    const weight = $('#weight_kg'), height = $('#height_cm'), bmiOut = $('[data-bmi]');
    const updateBmi = () => {
      const w = parseFloat(weight.value), h = parseFloat(height.value);
      if (!(w > 0 && h > 0)) { bmiOut.innerHTML = '<b>--</b><span>Enter weight and height</span>'; return; }
      const bmi = w / ((h / 100) ** 2);
      const band = bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Healthy weight' : bmi < 30 ? 'Overweight' : 'Obese';
      bmiOut.innerHTML = `<b>${bmi.toFixed(1)} kg/m²</b><span>${band}</span>`;
    };
    [weight, height].forEach((el) => el.addEventListener('input', updateBmi));
    updateBmi();

    /* menstrual pattern applies to female patients only */
    const femaleOnly = $('[data-female-only]');
    const syncSex = () => {
      const sex = $('input[name=sex]:checked', wizard)?.value;
      const radios = $$('input[name=menstrual_pattern]', wizard);
      const male = sex === 'Male';
      femaleOnly.hidden = male;
      radios.forEach((r) => { r.disabled = male; r.required = !male; });
    };
    $$('input[name=sex]', wizard).forEach((r) => r.addEventListener('change', syncSex));
    syncSex();

    /* mutually exclusive symptom pairs */
    $$('input[data-exclusive]', wizard).forEach((box) => box.addEventListener('change', () => {
      if (!box.checked) return;
      $$(`input[data-exclusive="${box.dataset.exclusive}"]`, wizard).forEach((other) => {
        if (other !== box) other.checked = false;
      });
    }));

    /* glucose unit switch converts limits and any value already typed */
    const glucose = $('#glucose'), unitLabel = $('[data-glucose-unit]');
    const K = 18.016;
    let unitNow = $('input[name=glucose_unit]:checked', wizard)?.value || 'mg/dL';
    const applyUnit = (unit, convert) => {
      const mmol = unit === 'mmol/L';
      if (convert && glucose.value) {
        const v = parseFloat(glucose.value);
        if (isFinite(v)) glucose.value = mmol ? (v / K).toFixed(1) : Math.round(v * K);
      }
      glucose.min = mmol ? (30 / K).toFixed(1) : 30;
      glucose.max = mmol ? (700 / K).toFixed(1) : 700;
      glucose.step = mmol ? 0.1 : 1;
      glucose.dataset.unitNow = unit;
      unitLabel.textContent = unit;
      unitNow = unit;
    };
    $$('input[name=glucose_unit]', wizard).forEach((r) => r.addEventListener('change', () => {
      if (r.checked && r.value !== unitNow) applyUnit(r.value, true);
    }));
    applyUnit(unitNow, false);

    /* open on the first step that has a server-side error */
    const firstError = $('.has-error', wizard);
    show(firstError ? Math.max(stepOf(firstError), 0) : 0, false);
    $$('[data-goto]').forEach((link) => link.addEventListener('click', (e) => {
      const target = document.getElementById(link.dataset.goto);
      if (!target) return;
      e.preventDefault();
      show(Math.max(stepOf(target), 0), false);
      (target.matches('input, select') ? target : target.querySelector('input'))?.focus();
    }));
  }

  /* ---- Admin: poll training status ---- */
  const poll = $('[data-poll]');
  if (poll && $('[data-status="training"]', poll)) {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(poll.dataset.poll, { headers: { Accept: 'application/json' } });
        if (!res.ok) return;
        const data = await res.json();
        let busy = false;
        Object.entries(data).forEach(([id, info]) => {
          const row = $(`[data-dataset="${id}"]`, poll);
          if (!row) return;
          $('[data-status-msg]', row).textContent = info.message;
          if (info.status === 'training') busy = true;
          else if (row.dataset.status === 'training') { clearInterval(timer); window.location.reload(); }
        });
        if (!busy) clearInterval(timer);
      } catch (err) { /* offline: try again on the next tick */ }
    }, 4000);
  }
})();
