PIL BUNKERING DECISION-SUPPORT ENGINE
========================================
MaritimeONE Case Summit 2026 -- PIL Challenge: Optimising Bunkering Strategy


HOW TO RUN IT
-------------
1. Make sure Python 3 is installed, with: numpy, pandas, matplotlib, scikit-learn
   (install with:  pip install numpy pandas matplotlib scikit-learn)

2. Run:
       python3 pil_bunkering_optimizer.py

3. That's it. It will:
     - run the full pipeline (ML price forecasting, DP optimisation,
       Monte Carlo stress testing)
     - print results to the terminal
     - save charts + a results.json into a new "output" folder created
       next to the script
     - automatically open the interactive dashboard in your default
       web browser, pre-loaded with the numbers it just computed

IMPORTANT: pil_bunkering_optimizer.py and pil_route_builder_dashboard.html
must stay in the SAME folder. The Python script reads the HTML file as a
template and writes a "live" copy with real computed data baked in -- if
it can't find the HTML file next to it, it'll skip the dashboard step and
just print a note, everything else still works.

You can also open pil_route_builder_dashboard.html directly (double-click,
no Python needed) -- it'll work standalone with static illustrative
reference prices instead of the ML-forecasted ones.


FOLDER CONTENTS
----------------
pil_bunkering_optimizer.py
    The full engine in one file: synthetic price-history generation, ML
    price forecasting (gradient-boosted trees), the dynamic-programming
    bunkering optimiser, baseline comparison strategies, Monte Carlo
    scenario testing, and the dashboard auto-launcher. Fully commented,
    organised into 6 clearly labelled sections.

pil_route_builder_dashboard.html
    The interactive tool. A single self-contained file -- no build step,
    no npm, no internet connection required. Build any route by clicking
    ports on a real 3D globe (actual coastlines, drag to rotate), compare
    optimised vs. price-blind bunkering costs, run price-spike / port-
    closure stress tests, and edit port prices or vessel specs directly
    in the browser.

Presentation_Deck/
    PIL_Bunkering_POC.pptx -- the 12-page proof-of-concept deck (problem
    statement, methodology, results, scenario testing, business impact,
    implementation roadmap), built to fit the case brief's submission
    format.

Charts/
    The four charts from the deck as standalone PNGs, in case you want
    them for other slides or documents.

Sample_Outputs/
    Pre-generated example outputs, so you can see results without running
    anything:
      - results.json               -- full results from the SWS demo run
      - custom_route_results.json  -- example custom-route output
      - pil_dashboard_live.html    -- example of what the auto-launched
                                       "live" dashboard looks like

Legacy_React_Versions/
    Earlier React (.jsx) versions of the dashboard, built before the
    final standalone HTML version. Kept for reference / in case you want
    to integrate this into a React codebase later, but the .html file at
    the project root is the current, recommended version -- it does
    everything these do, plus the 3D globe and the Python integration,
    with no build tools required.


A NOTE ON THE DATA
-------------------
No API keys or paid data feeds were used anywhere in this project. Port
coordinates are real; world coastline shapes are real (pulled once from a
public Natural Earth mirror on GitHub, no auth required). Bunker fuel
prices and their historical time series are illustrative/synthetic --
there is no live pricing feed behind this. The forecasting model and
optimisation algorithm are both genuinely functional; swapping in real
price data would be a data-integration step, not a rebuild. See the
Assumptions section of the deck and the comments in
pil_bunkering_optimizer.py for the full breakdown of what's real vs.
illustrative.
