const videoGroups = [
  {
    id: "deprl",
    title: "DepRL",
    description: "Full benchmark coverage from the exploration-enhanced reward-based baseline.",
    files: [
      "balance", "catch", "chinup", "crawl", "door", "hurdle", "jump", "polewalk", "powerlift", "reach", "run",
      "sidestep", "sit", "slide", "squat", "stair", "stance", "stand", "steppingstones", "walk", "walkandsit", "walkturn",
    ],
  },
  {
    id: "sac",
    title: "SAC",
    description: "Soft actor-critic rollouts across the full task set.",
    files: [
      "balance", "catch", "chinup", "crawl", "door", "hurdle", "jump", "polewalk", "powerlift", "reach", "run",
      "sidestep", "sit", "slide", "squat", "stair", "stance", "stand", "stepstones", "walk", "walkandsit", "walkturn",
    ],
  },
  {
    id: "ppo",
    title: "PPO",
    description: "On-policy reinforcement learning rollouts for the full task set.",
    files: [
      "balance", "catch", "chinup", "crawl", "door", "hurdle", "jump", "polewalk", "powerlift", "reach", "run",
      "sidestep", "sit", "slide", "squat", "stair", "stance", "stand", "steppingstones", "walk", "walkandsit", "walkturn",
    ],
  },
  {
    id: "dynsyn-sac",
    title: "DynSyn-SAC",
    description: "Dynamic-synergy SAC rollouts across the full task set.",
    files: [
      "balance", "catch", "chinup", "crawl", "door", "hurdle", "jump", "polewalk", "powerlift", "reach", "run",
      "sidestep", "sit", "slide", "squat", "stair", "stance", "stand", "steppingstones", "walk", "walkandsit", "walkturn",
    ],
  },
  {
    id: "musclemimic",
    title: "MuscleMimic",
    description: "Imitation-prior policy examples for reference-motion control.",
    files: ["jump", "run", "stair", "stand", "walk"],
  },
  {
    id: "residual-rl",
    title: "Residual RL",
    description: "Residual adaptation examples used for walk, run, and stairs.",
    files: ["run", "stair", "walk"],
  },
];

const taskLabels = new Map([
  ["balance", "balance"],
  ["catch", "catch"],
  ["chinup", "chin up"],
  ["crawl", "crawl"],
  ["door", "open door"],
  ["hurdle", "hurdle"],
  ["jump", "jump"],
  ["polewalk", "pole walk"],
  ["powerlift", "powerlift"],
  ["reach", "reach"],
  ["run", "run"],
  ["sidestep", "sidestep"],
  ["sit", "sit"],
  ["slide", "slide"],
  ["squat", "squat"],
  ["stair", "stairs"],
  ["stance", "singleleg stand"],
  ["stand", "stand"],
  ["steppingstones", "stepping stones"],
  ["stepstones", "stepping stones"],
  ["walk", "walk forward"],
  ["walkandsit", "walk and sit"],
  ["walkturn", "walk turn"],
]);

const groupTone = new Map([
  ["deprl", "gallery-tone-teal"],
  ["sac", "gallery-tone-blue"],
  ["ppo", "gallery-tone-coral"],
  ["dynsyn-sac", "gallery-tone-gold"],
  ["musclemimic", "gallery-tone-green"],
  ["residual-rl", "gallery-tone-indigo"],
]);

const taskFamilies = [
  {
    id: "stabilization",
    title: "Stabilization",
    description: "Postural regulation and fundamental strength skills.",
    tasks: [
      { keys: ["stand"], label: "stand" },
      { keys: ["powerlift"], label: "powerlift" },
      { keys: ["squat"], label: "squat" },
      { keys: ["sit"], label: "sit" },
      { keys: ["stance"], label: "singleleg stand" },
      { keys: ["balance"], label: "balance" },
    ],
  },
  {
    id: "locomotion",
    title: "Locomotion",
    description: "Locomotor skills across different gaits and dynamic behaviors.",
    tasks: [
      { keys: ["walk"], label: "walk forward" },
      { keys: ["run"], label: "run" },
      { keys: ["walkturn"], label: "walk turn" },
      { keys: ["jump"], label: "jump" },
      { keys: ["sidestep"], label: "sidestep" },
      { keys: ["crawl"], label: "crawl" },
    ],
  },
  {
    id: "interaction",
    title: "Interaction",
    description: "Whole-body interaction with terrain, objects, and external constraints.",
    tasks: [
      { keys: ["stair"], label: "stairs" },
      { keys: ["catch"], label: "catch" },
      { keys: ["hurdle"], label: "hurdle" },
      { keys: ["slide"], label: "slide" },
      { keys: ["steppingstones", "stepstones"], label: "stepping stones" },
      { keys: ["polewalk"], label: "pole walk" },
      { keys: ["reach"], label: "reach" },
      { keys: ["walkandsit"], label: "walk and sit" },
      { keys: ["chinup"], label: "chin up" },
      { keys: ["door"], label: "open door" },
    ],
  },
];

const taskPolicyViews = [
  { id: "deprl", title: "DepRL", groups: ["deprl"] },
  { id: "sac", title: "SAC", groups: ["sac"] },
  { id: "ppo", title: "PPO", groups: ["ppo"] },
  { id: "dynsyn-sac", title: "DynSyn-SAC", groups: ["dynsyn-sac"] },
  {
    id: "imitation-prior",
    title: "Imitation-prior control",
    groups: ["musclemimic"],
    note: "Due to missing reference motions, the other tasks either cannot be evaluated or failed during evaluation.",
  },
  { id: "residual-adaptation", title: "Residual adaptation over imitation priors", groups: ["residual-rl"] },
];

const familyPolicySelections = new Map();

function getGroupById(groupId) {
  return videoGroups.find((group) => group.id === groupId);
}

function getPolicyViewById(viewId) {
  return taskPolicyViews.find((view) => view.id === viewId) ?? taskPolicyViews[0];
}

function findTaskFile(group, task) {
  if (!group) return null;
  return task.keys.find((key) => group.files.includes(key)) ?? null;
}

function getPolicyVideosByFamily(view, family) {
  return view.groups.flatMap((groupId) => {
    const group = getGroupById(groupId);
    if (!group) return [];

    return family.tasks.map((task) => {
      const file = findTaskFile(group, task);
      if (!file) return null;

      return {
        group,
        file,
        label: task.label,
        src: `./static/videos/gallery/${group.id}/${file}.mp4`,
      };
    }).filter(Boolean);
  });
}

function countFamilyPolicyVideos(view, family) {
  return getPolicyVideosByFamily(view, family).length;
}

function getFamilyPolicyView(family) {
  const selectedId = familyPolicySelections.get(family.id) ?? taskPolicyViews[0].id;
  return getPolicyViewById(selectedId);
}

function setAllFamilyPolicyViews(viewId) {
  const view = getPolicyViewById(viewId);
  taskFamilies.forEach((family) => {
    familyPolicySelections.set(family.id, view.id);
  });
  return view;
}

function setupLearningCurveLegend() {
  const figure = document.getElementById("learning-curve-figure");
  if (!figure || figure.dataset.legendReady) return;

  figure.querySelectorAll("[data-linked-policy]").forEach((button) => {
    button.addEventListener("click", () => {
      renderTaskFamilyBrowser(button.dataset.linkedPolicy);
      document.getElementById("task-family-browser")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });

  figure.dataset.legendReady = "true";
}

function updateLearningCurveLegend() {
  const figure = document.getElementById("learning-curve-figure");
  if (!figure) return;

  const selectedViews = taskFamilies.map((family) => getFamilyPolicyView(family));
  const firstView = selectedViews[0] ?? taskPolicyViews[0];
  const allFamiliesMatch = selectedViews.every((view) => view.id === firstView.id);
  figure.dataset.activePolicy = allFamiliesMatch ? firstView.id : "mixed";

  const linkedButtons = figure.querySelectorAll("[data-linked-policy]");
  let activeLabel = "";

  linkedButtons.forEach((button) => {
    const selected = allFamiliesMatch && button.dataset.linkedPolicy === firstView.id;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
    if (selected) activeLabel = button.textContent.trim();
  });

  const note = document.getElementById("curve-link-note");
  if (!note) return;

  if (!allFamiliesMatch) {
    note.textContent = "Each task family above has its own selected policy; use the chart legend to synchronize the comparable curve methods.";
    return;
  }

  note.textContent = activeLabel
    ? `${activeLabel} curves highlighted; task videos above are grouped by ${firstView.title}.`
    : `${firstView.title} videos above are shown separately; this learning-curve figure compares DepRL, DynSyn-SAC, SAC, and PPO.`;
}

function renderFamilyPolicyTabs(family, activeView) {
  return taskPolicyViews.map((view) => {
    const selected = view.id === activeView.id;
    const count = countFamilyPolicyVideos(view, family);
    return `
      <button class="policy-tab family-policy-tab${selected ? " is-active" : ""}" type="button" role="tab" aria-selected="${selected}" data-family-id="${family.id}" data-policy-view="${view.id}">
        <span>${view.title}</span>
        <small>${count} videos</small>
      </button>
    `;
  }).join("");
}

function renderTaskFamilyBrowser(syncViewId) {
  const browser = document.getElementById("task-family-browser");
  if (!browser) return;

  setupLearningCurveLegend();
  if (syncViewId) setAllFamilyPolicyViews(syncViewId);

  browser.innerHTML = taskFamilies.map((family) => {
    const activeView = getFamilyPolicyView(family);
    const videos = getPolicyVideosByFamily(activeView, family);
    const note = activeView.note ? `<p class="policy-empty">${activeView.note}</p>` : "";
    const cards = videos.map((item) => `
      <figure class="policy-video-card gallery-card">
        <video src="${item.src}" controls autoplay muted loop playsinline preload="metadata" aria-label="${item.group.title} ${item.label} video"></video>
        <figcaption>
          <span>${item.label}</span>
          <small>${item.group.title}</small>
        </figcaption>
      </figure>
    `).join("");

    const empty = `<p class="policy-empty">No ${activeView.title} videos available for this task family.</p>`;

    return `
      <article class="family ${family.id}">
        <div class="family-header">
          <h3>${family.title}</h3>
          <p>${family.description}</p>
          <span class="family-count">${videos.length} videos</span>
        </div>
        <div class="family-policy-tabs" role="tablist" aria-label="${family.title} control method selector">
          ${renderFamilyPolicyTabs(family, activeView)}
        </div>
        ${note}
        <div class="thumb-grid policy-thumb-grid${family.id === "interaction" ? " wide" : ""}">
          ${videos.length ? cards : empty}
        </div>
      </article>
    `;
  }).join("");

  browser.querySelectorAll("[data-family-id][data-policy-view]").forEach((button) => {
    button.addEventListener("click", () => {
      familyPolicySelections.set(button.dataset.familyId, button.dataset.policyView);
      renderTaskFamilyBrowser();
    });
  });

  updateLearningCurveLegend();
}

function renderVideoGallery() {
  const summary = document.getElementById("video-gallery-summary");
  const gallery = document.getElementById("video-gallery");
  if (!summary || !gallery) return;

  const total = videoGroups.reduce((count, group) => count + group.files.length, 0);
  summary.innerHTML = [
    `<article><strong>${total}</strong><span>total videos</span></article>`,
    `<article><strong>${videoGroups.length}</strong><span>policy families</span></article>`,
    `<article><strong>22</strong><span>maximum tasks per full baseline</span></article>`,
  ].join("");

  gallery.innerHTML = videoGroups.map((group) => {
    const tone = groupTone.get(group.id) ?? "gallery-tone-teal";
    const videos = group.files.map((file) => {
      const label = taskLabels.get(file) ?? file;
      return `
        <figure class="gallery-card">
          <video src="./static/videos/gallery/${group.id}/${file}.mp4" controls muted playsinline preload="metadata" aria-label="${group.title} ${label} video"></video>
          <figcaption>
            <span>${label}</span>
            <small>${group.title}</small>
          </figcaption>
        </figure>
      `;
    }).join("");

    return `
      <details class="video-group ${tone}" open>
        <summary>
          <span>
            <strong>${group.title}</strong>
            <em>${group.description}</em>
          </span>
          <b>${group.files.length} videos</b>
        </summary>
        <div class="gallery-grid">${videos}</div>
      </details>
    `;
  }).join("");
}

renderTaskFamilyBrowser();
renderVideoGallery();
