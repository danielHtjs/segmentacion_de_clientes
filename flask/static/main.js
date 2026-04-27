// ── Paleta de colores por cluster ────────────────────────────────
const COLORS = [
  "#7c6fff","#43d9ad","#ffb347","#ff6584",
  "#7ec8e3","#c7ceea","#b5ead7","#ffd3b6"
];

// ── Referencias DOM ──────────────────────────────────────────────
const mallSelect  = document.getElementById("mall-select");
const kSlider     = document.getElementById("k-slider");
const kDisplay    = document.getElementById("k-display");
const btnAnalizar = document.getElementById("btn-analizar");
const btnText     = document.getElementById("btn-text");
const btnLoader   = document.getElementById("btn-loader");

// Métricas header
const mSil = document.getElementById("m-sil");
const mK   = document.getElementById("m-k");
const mN   = document.getElementById("m-n");
const pageSub = document.getElementById("page-sub");

// Estado inicial / panels
const estadoInicial = document.getElementById("estado-inicial");

// Charts activos (para destruir antes de redibujar)
let chartPCA     = null;
let chartInertia = null;
let chartSil     = null;

// ── Slider live update ───────────────────────────────────────────
kSlider.addEventListener("input", () => {
  kDisplay.textContent = kSlider.value;
});

// ── Tabs ─────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    // Solo activa si hay datos
    if (estadoInicial.classList.contains("hidden") === false) return;

    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));
    document.getElementById(btn.dataset.panel).classList.remove("hidden");
  });
});

// ── Llamada al backend ───────────────────────────────────────────
btnAnalizar.addEventListener("click", async () => {
  const mall = mallSelect.value;
  const k    = parseInt(kSlider.value);

  // Estado loading
  btnAnalizar.disabled = true;
  btnText.textContent  = "Analizando…";
  btnLoader.classList.remove("hidden");

  try {
    const res  = await fetch("/api/segmentar", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ mall, k }),
    });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error);

    renderizarTodo(data);

  } catch (err) {
    alert("Error al conectar con el servidor: " + err.message);
  } finally {
    btnAnalizar.disabled = false;
    btnText.textContent  = "Analizar";
    btnLoader.classList.add("hidden");
  }
});

// ── Render principal ─────────────────────────────────────────────
function renderizarTodo(data) {
  // Ocultar estado inicial, mostrar primer panel
  estadoInicial.classList.add("hidden");
  document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));
  document.getElementById("panel-pca").classList.remove("hidden");

  // Activar primera tab
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelector('[data-panel="panel-pca"]').classList.add("active");

  // Métricas header
  const sil = data.silhouette;
  mSil.textContent = sil.toFixed(4);
  mSil.style.color = sil >= 0.40 ? "var(--accent2)" : "var(--danger)";
  mK.textContent   = data.k;
  mN.textContent   = data.n_total.toLocaleString();
  pageSub.textContent = `${mallSelect.value} · ${data.n_total.toLocaleString()} transacciones · k = ${data.k}`;

  // Dibujar charts
  dibujarPCA(data.scatter, data.k);
  dibujarElbow(data.elbow, data.k);
  dibujarPerfiles(data.perfiles);
}

// ── PCA Scatter ──────────────────────────────────────────────────
function dibujarPCA(scatter, k) {
  if (chartPCA) chartPCA.destroy();

  // Agrupar por cluster
  const datasets = [];
  for (let c = 0; c < k; c++) {
    const pts = scatter.filter(p => p.cluster === c);
    datasets.push({
      label:           `Segmento ${c + 1}`,
      data:            pts.map(p => ({ x: p.x, y: p.y, age: p.age, spend: p.spend, cat: p.category, gender: p.gender })),
      backgroundColor: COLORS[c] + "99",
      pointRadius:     3,
      pointHoverRadius:6,
    });
  }

  chartPCA = new Chart(document.getElementById("chart-pca"), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: {
          labels: { color: "#7070a0", font: { family: "'DM Mono'" }, boxWidth: 10, boxHeight: 10 }
        },
        tooltip: {
          callbacks: {
            label: ctx => {
              const p = ctx.raw;
              return [`Segmento ${ctx.datasetIndex + 1}`, `Edad: ${p.age}`, `Gasto: $${p.spend.toLocaleString()}`, `Cat: ${p.cat}`, p.gender];
            }
          },
          backgroundColor: "#181830",
          borderColor: "rgba(255,255,255,0.08)",
          borderWidth: 1,
        }
      },
      scales: {
        x: { ticks: { color: "#7070a0" }, grid: { color: "rgba(255,255,255,0.04)" }, title: { display: true, text: "Componente 1", color: "#7070a0" } },
        y: { ticks: { color: "#7070a0" }, grid: { color: "rgba(255,255,255,0.04)" }, title: { display: true, text: "Componente 2", color: "#7070a0" } },
      }
    }
  });
}

// ── Elbow ────────────────────────────────────────────────────────
function dibujarElbow(elbow, kSel) {
  if (chartInertia) chartInertia.destroy();
  if (chartSil)     chartSil.destroy();

  const ks       = elbow.map(e => e.k);
  const inertias = elbow.map(e => e.inertia);
  const sils     = elbow.map(e => e.silhouette);

  const baseOpts = (yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
      legend: { display: false },
      tooltip: { backgroundColor: "#181830", borderColor: "rgba(255,255,255,0.08)", borderWidth: 1, titleColor: "#e8e8f0", bodyColor: "#7070a0" }
    },
    scales: {
      x: { ticks: { color: "#7070a0", font: { family: "'DM Mono'" } }, grid: { color: "rgba(255,255,255,0.04)" }, title: { display: true, text: "k", color: "#7070a0" } },
      y: { ticks: { color: "#7070a0", font: { family: "'DM Mono'" } }, grid: { color: "rgba(255,255,255,0.04)" }, title: { display: true, text: yLabel, color: "#7070a0" } },
    }
  });

  const annotLine = (value, color) => ({
    type: "line", xMin: value - 1, xMax: value - 1,
    borderColor: color, borderWidth: 2, borderDash: [4, 4]
  });

  chartInertia = new Chart(document.getElementById("chart-inertia"), {
    type: "line",
    data: {
      labels: ks,
      datasets: [{
        data: inertias, borderColor: "#7c6fff", backgroundColor: "#7c6fff22",
        fill: true, tension: 0.3, pointBackgroundColor: ks.map(k => k === kSel ? "#7c6fff" : "#181830"),
        pointBorderColor: "#7c6fff", pointRadius: 5,
      }]
    },
    options: { ...baseOpts("Inercia (WCSS)"), animation: { duration: 400 } }
  });

  chartSil = new Chart(document.getElementById("chart-sil"), {
    type: "line",
    data: {
      labels: ks,
      datasets: [
        {
          label: "Silhouette", data: sils, borderColor: "#43d9ad", backgroundColor: "#43d9ad22",
          fill: true, tension: 0.3,
          pointBackgroundColor: ks.map(k => k === kSel ? "#43d9ad" : "#181830"),
          pointBorderColor: "#43d9ad", pointRadius: 5,
        },
        {
          label: "Objetivo 0.40", data: ks.map(() => 0.40),
          borderColor: "#ff658466", borderDash: [6, 4], borderWidth: 1.5,
          pointRadius: 0, fill: false,
        }
      ]
    },
    options: {
      ...baseOpts("Silhouette Score"),
      plugins: {
        legend: {
          display: true,
          labels: { color: "#7070a0", font: { family: "'DM Mono'", size: 11 }, boxWidth: 12, boxHeight: 12 }
        }
      }
    }
  });
}

// ── Perfiles — cards ─────────────────────────────────────────────
function dibujarPerfiles(perfiles) {
  const container = document.getElementById("cards-container");
  container.innerHTML = "";

  perfiles.forEach((p, i) => {
    const color = COLORS[i % COLORS.length];
    const rows = [
      ["Edad promedio",    `${p.edad_prom} años`],
      ["Gasto promedio",   `$${p.gasto_prom.toLocaleString()}`],
      ["Cantidad promedio",p.cantidad_prom],
      ["Categoría top",   p.top_cat],
      ["% Femenino",      `${p.pct_fem}%`],
      ["Pago preferido",  p.top_pago],
      ["Transacciones",   p.n.toLocaleString()],
    ];

    const card = document.createElement("div");
    card.className = "seg-card";
    card.style.borderTopColor = color;
    card.innerHTML = `
      <p class="seg-card-title" style="color:${color}">Segmento ${p.cluster + 1}</p>
      <p class="seg-card-pct">${p.pct}% del total</p>
      ${rows.map(([k,v]) => `
        <div class="seg-row">
          <span class="seg-key">${k}</span>
          <span class="seg-val">${v}</span>
        </div>`).join("")}
    `;
    container.appendChild(card);
  });
}
