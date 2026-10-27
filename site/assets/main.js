/* ANTEVRIC — interacciones. JS puro, sin dependencias. */
(function () {
  "use strict";

  var doc = document.documentElement;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Carga inicial: dispara la entrada del titular */
  requestAnimationFrame(function () {
    requestAnimationFrame(function () { document.body.classList.add("is-loaded"); });
  });

  /* Navegación móvil */
  var toggle = document.querySelector(".nav-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var open = document.body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.querySelector(".t").textContent = open ? "Cerrar" : "Menú";
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.classList.contains("nav-open")) toggle.click();
    });
  }

  /* Hora local de México en la barra superior */
  var clock = document.querySelector("[data-clock]");
  if (clock) {
    var fmt = new Intl.DateTimeFormat("es-MX", {
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: "America/Mexico_City"
    });
    var tick = function () { clock.textContent = fmt.format(new Date()); };
    tick();
    setInterval(tick, 1000);
  }

  /* Año en el pie */
  document.querySelectorAll("[data-year]").forEach(function (el) { el.textContent = new Date().getFullYear(); });

  /* Aparición al hacer scroll */
  var revealables = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    revealables.forEach(function (el) { io.observe(el); });
  } else {
    revealables.forEach(function (el) { el.classList.add("is-in"); });
  }

  /* Pila de infraestructura del hero */
  var stack = document.querySelector("[data-stack]");
  if (stack) {
    var layers = Array.prototype.slice.call(stack.querySelectorAll("li"));
    var readout = document.querySelector("[data-stack-readout]");
    var readLabel = document.querySelector("[data-stack-label]");
    var idx = layers.length - 1;
    var timer = null;

    var setActive = function (i) {
      layers.forEach(function (l, j) { l.classList.toggle("is-active", j === i); });
      if (readout) readout.textContent = layers[i].getAttribute("data-desc");
      if (readLabel) readLabel.textContent = layers[i].getAttribute("data-key") + " · " + layers[i].getAttribute("data-name");
      idx = i;
    };
    var cycle = function () { setActive(idx - 1 < 0 ? layers.length - 1 : idx - 1); };
    var start = function () { if (!reduce) { stop(); timer = setInterval(cycle, 2600); } };
    var stop = function () { if (timer) clearInterval(timer); timer = null; };

    setTimeout(function () { stack.classList.add("is-built"); setActive(idx); start(); }, reduce ? 0 : 450);

    layers.forEach(function (l, i) {
      l.setAttribute("tabindex", "0");
      l.addEventListener("mouseenter", function () { stop(); setActive(i); });
      l.addEventListener("focus", function () { stop(); setActive(i); });
    });
    stack.addEventListener("mouseleave", start);
    stack.addEventListener("focusout", start);
  }

  /* Índice lateral de servicios */
  var index = document.querySelector("[data-svc-index]");
  if (index && "IntersectionObserver" in window) {
    var links = index.querySelectorAll("a");
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (a) { a.classList.remove("is-active"); });
          var a = map[en.target.id];
          if (a) a.classList.add("is-active");
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll(".svc[id]").forEach(function (s) { spy.observe(s); });
  }

  /* Formulario: preselecciona el área si viene en la URL (?area=...) */
  var area = new URLSearchParams(window.location.search).get("area");
  if (area) {
    var opt = document.querySelector('input[name="area"][value="' + area.replace(/"/g, "") + '"]');
    if (opt) opt.checked = true;
  }
})();
