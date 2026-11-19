"use strict";
// Credit card default risk - the web page.
// 1. Builds the form from model/model_info.json (the 23 inputs, their codes and labels).
// 2. Loads Python in the browser (Pyodide), then predictor.py and the exported CatBoost model.
// 3. Re-checks the three examples against their Module 8 risk scores before enabling the form.
// 4. Sends the form values to predictor.predict_json() and shows the risk and the decision.

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const APP_DIR = "/home/pyodide/app"; // folder inside Pyodide's in-memory file system

const state = { info: null, predictor: null, example: null };
const $ = (id) => document.getElementById(id);

function setStatus(text, kind) {
  const status = $("status");
  status.textContent = text;
  status.dataset.state = kind;
}

async function fetchText(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`could not download ${path} (HTTP ${response.status})`);
  return response.text();
}

const percent = (x, digits = 1) => `${(100 * x).toFixed(digits)}%`;
const decimal = (x) => x.toFixed(3).replace(/^0/, ""); // .524, the style used in the reports
const count = (n) => n.toLocaleString("en-US");
const sortedCodes = (codes) => Object.keys(codes).sort((a, b) => Number(a) - Number(b));

function fieldLabel(f) {
  // In the month groups the legend already says what the numbers are, so only the month is shown.
  if (f.group !== "client") return f.label.split(", ")[1];
  return f.unit ? `${f.label} (${f.unit})` : f.label;
}

function buildForm(info) {
  const holder = $("fields");
  for (const [group, title] of Object.entries(info.groups)) {
    const fieldset = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = title;
    const grid = document.createElement("div");
    grid.className = "grid";
    for (const f of info.features.filter((x) => x.group === group)) {
      const field = document.createElement("div");
      field.className = "field";
      const label = document.createElement("label");
      label.htmlFor = `f-${f.name}`;
      label.textContent = fieldLabel(f);
      let input;
      if (f.codes) {
        input = document.createElement("select");
        for (const code of sortedCodes(f.codes)) input.add(new Option(f.codes[code], code));
      } else {
        input = document.createElement("input");
        input.type = "text";
        input.autocomplete = "off";
        input.spellcheck = false;
        // Bills can be negative (a credit balance), so they get the full keyboard on phones.
        input.inputMode = f.valid.min === undefined ? "text" : f.whole ? "numeric" : "decimal";
      }
      input.id = `f-${f.name}`;
      input.name = f.name;
      input.addEventListener("input", () => {
        state.example = null; // the values no longer match an example exactly
        markExample();
      });
      field.append(label, input);
      grid.append(field);
    }
    fieldset.append(legend, grid);
    holder.append(fieldset);
  }
}

function fillForm(values) {
  for (const f of state.info.features) {
    const input = $(`f-${f.name}`);
    const value = values[f.name];
    input.value = !f.codes && f.unit === "NT$" ? Number(value).toLocaleString("en-US") : String(value);
    input.removeAttribute("aria-invalid");
  }
}

function readForm() {
  const record = {};
  for (const f of state.info.features) {
    let value = $(`f-${f.name}`).value;
    // Accept "20,000" and a typographic minus sign; predictor.py does all the other checks.
    if (!f.codes) value = value.replace(/[\s,]/g, "").replace(/−/g, "-");
    record[f.name] = value;
  }
  return record;
}

function buildExamples(info) {
  const holder = $("example-buttons");
  info.examples.forEach((example, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.label;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      fillForm(example.values);
      state.example = i;
      markExample();
      if (state.predictor) estimate();
    });
    holder.append(button);
  });
}

function markExample() {
  [...$("example-buttons").children].forEach((button, i) => {
    button.setAttribute("aria-pressed", String(i === state.example));
  });
}

function addParagraph(parent, title, text) {
  const heading = document.createElement("h3");
  heading.textContent = title;
  const p = document.createElement("p");
  p.append(...(Array.isArray(text) ? text : [text]));
  parent.append(heading, p);
}

function buildAbout(info) {
  const m = info.model, r = info.test_results, t = r.at_threshold, h = r["at_0.5"], d = info.data;
  const about = $("about");
  addParagraph(about, "Model", `CatBoost (gradient-boosted decision trees), tuned in Module 8: ${m.trees} trees of ` +
    `depth ${m.depth}, learning rate ${m.learning_rate}, L2 penalty ${m.l2_leaf_reg}. It learned from ` +
    `${count(m.learned_from_clients)} training clients; another ${count(m.early_stopping_clients)} were held back ` +
    `to decide when to stop adding trees (early stopping kept ${m.trees} of the ${m.trees_built} trees built).`);
  addParagraph(about, "Threshold", `A client is flagged when the estimated risk is ${percent(info.threshold)} ` +
    `(${info.threshold.toFixed(4)}) or more. Module 8 chose this threshold because it gave the highest F1 score ` +
    `on out-of-fold predictions for the ${count(d.train_clients)} training clients; the test clients were not ` +
    `used to choose it.`);
  addParagraph(about, "Test results", `On the ${count(r.clients)} test clients, whom the model never saw in ` +
    `training (${count(r.defaulters)} of them defaulted): ROC-AUC ${decimal(r.roc_auc)}. At the ` +
    `${percent(info.threshold)} threshold: precision ${decimal(t.precision)}, recall ${decimal(t.recall)}, ` +
    `F1 ${decimal(t.f1)}. At 50%: precision ${decimal(h.precision)}, recall ${decimal(h.recall)}, F1 ${decimal(h.f1)}.`);
  addParagraph(about, "How this page works", `CatBoost exported the trained model as plain Python code. ` +
    `Pyodide runs that code, together with predictor.py, inside your browser, so no server receives your inputs. ` +
    `Before each deployment, automated tests in GitHub Actions check that this code reproduces the Module 8 risk ` +
    `score of every one of the ${count(r.clients)} test clients, and each time the page loads it re-checks the ` +
    `three examples.`);
  const link = document.createElement("a");
  link.href = d.doi;
  link.textContent = d.doi;
  addParagraph(about, "Data", [`Yeh, I. (2009). Default of Credit Card Clients [Dataset]. ${d.source}. `, link,
    `. License: CC BY 4.0. ${count(d.clients)} credit card clients in Taiwan, ${d.period}, split at random into ` +
    `${count(d.train_clients)} training and ${count(d.test_clients)} test clients (seed 20).`]);
  addParagraph(about, "Data cleaning", "As before training, the undocumented education codes 0, 5 and 6 count " +
    "as 4 (others), and the undocumented marital status code 0 counts as 3 (others). The undocumented " +
    "repayment status codes −2 and 0 are kept as they are, as in training.");
  addParagraph(about, "Limits", "The model learned from one Taiwanese bank's clients in 2005, so it may not suit " +
    "other lenders or later years. Its inputs include sex, age and marital status, which a real lender would " +
    "have to review for fairness and legal compliance.");
}

function showProblems(problems) {
  const box = $("problems");
  const intro = document.createElement("p");
  intro.textContent = problems.length === 1 ? "Please fix this input:" : `Please fix these ${problems.length} inputs:`;
  const list = document.createElement("ul");
  for (const text of problems) {
    const item = document.createElement("li");
    item.textContent = text;
    list.append(item);
    const f = state.info.features.find((x) => text.startsWith(`${x.label}:`));
    if (f) $(`f-${f.name}`).setAttribute("aria-invalid", "true");
  }
  box.replaceChildren(intro, list);
  box.hidden = false;
}

function showResult(result) {
  const info = state.info;
  $("risk-value").textContent = percent(result.risk);
  $("meter-fill").style.width = percent(result.risk, 2);
  $("meter-mark").style.left = percent(info.threshold, 2);
  $("meter-mark-label").textContent = `threshold ${percent(info.threshold)}`;
  const decision = $("decision");
  decision.dataset.flagged = String(result.flagged);
  decision.textContent = result.flagged
    ? `Flag for review: the estimated risk is at or above the ${percent(info.threshold)} threshold.`
    : `No flag: the estimated risk is below the ${percent(info.threshold)} threshold.`;
  const notes = $("notes");
  notes.replaceChildren(...result.notes.map((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
  notes.hidden = result.notes.length === 0;
  const outcome = $("example-outcome");
  outcome.hidden = state.example === null;
  if (state.example !== null) {
    const example = info.examples[state.example];
    outcome.textContent = `This is test client ${example.test_client_id}, who in fact ` +
      `${example.defaulted ? "defaulted" : "did not default"} the next month.`;
  }
  const r = info.test_results, t = r.at_threshold;
  $("context").textContent = `How good is this threshold? Among the ${count(r.clients)} test clients it flagged ` +
    `${count(t.flagged)}; ${count(t.TP)} of them (${percent(t.precision)}) did default, and they were ` +
    `${percent(t.recall)} of all ${count(r.defaulters)} clients who defaulted.`;
  $("result").hidden = false;
  $("result-title").focus();
}

function estimate() {
  const result = JSON.parse(state.predictor.predict_json(JSON.stringify(readForm())));
  for (const f of state.info.features) $(`f-${f.name}`).removeAttribute("aria-invalid");
  if (!result.ok) {
    $("result").hidden = true;
    showProblems(result.problems);
    return;
  }
  $("problems").hidden = true;
  showResult(result);
}

function selfCheck(info) {
  for (const example of info.examples) {
    const result = JSON.parse(state.predictor.predict_json(JSON.stringify(example.values)));
    if (!result.ok || Math.abs(result.risk - example.module8_risk) > 1e-12) {
      throw new Error(`self-check failed for test client ${example.test_client_id}`);
    }
  }
}

async function start() {
  const started = performance.now();
  $("client-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.predictor) estimate();
  });
  try {
    const infoText = await fetchText("model/model_info.json", { cache: "no-cache" });
    const info = JSON.parse(infoText);
    state.info = info;
    buildForm(info);
    buildExamples(info);
    buildAbout(info);
    fillForm(info.examples[0].values);
    state.example = 0;
    markExample();
    if (typeof loadPyodide !== "function") throw new Error("the Pyodide script did not load");
    const [pyodide, code, model] = await Promise.all([
      loadPyodide({ indexURL: PYODIDE_INDEX }),
      fetchText("predictor.py", { cache: "no-cache" }),
      // The hash in the name makes browsers fetch the model again whenever it changes.
      fetchText(`model/catboost_model.py?v=${info.model.exported_sha256.slice(0, 12)}`),
    ]);
    pyodide.FS.mkdirTree(`${APP_DIR}/model`);
    pyodide.FS.writeFile(`${APP_DIR}/predictor.py`, code);
    pyodide.FS.writeFile(`${APP_DIR}/model/model_info.json`, infoText);
    pyodide.FS.writeFile(`${APP_DIR}/model/catboost_model.py`, model);
    pyodide.runPython(`import sys\nsys.path.insert(0, "${APP_DIR}")`);
    state.predictor = pyodide.pyimport("predictor");
    selfCheck(info);
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    setStatus(`Model ready (loaded in ${seconds} s). Self-check passed: the three examples match Module 8.`, "ready");
    $("estimate").disabled = false;
    $("estimate-hint").textContent = "The calculation runs in your browser.";
  } catch (err) {
    console.error(err);
    setStatus(`The model could not be loaded: ${err.message}. Please reload the page.`, "error");
    $("estimate-hint").textContent = "Unavailable because the model did not load.";
  }
}

start();
