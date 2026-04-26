# Wisteria: An Intelligence-Driven Property & PropTech Platform
[![Status](https://img.shields.io/badge/status-in%20development-green)](https://github.com/brandonhenson/wisteria)

**Wisteria is an advanced property listings portal designed to move beyond the static data tables of traditional platforms like Rightmove or Zoopla. By fusing multi-modal property data with sophisticated AI, Wisteria provides automated valuations, deep qualitative insights, and a truly personalised user experience, aiming to eliminate buyer "analysis paralysis" and build institutional-grade trust.**

---

### Table of Contents
1.  [The Core Problem](#the-core-problem)
2.  [Key Features](#key-features)
3.  [Methodology: A Deep Dive](#methodology-a-deep-dive)
4.  [Data Sources](#data-sources)
5.[Performance & Limitations](#performance--limitations)
6.  [Plans for the future](#plans-for-the-future)

---

### The Core Problem
Traditional property portals present data as disconnected, static tables. This approach burdens the user with immense cognitive load, leading to:
*   **Analysis Paralysis:** Users are forced to manually compare dozens of listings across spreadsheets.
*   **Opaque Valuations:** Simple price estimates lack transparency, eroding user trust.
*   **Qualitative Blind Spots:** Key lifestyle and aesthetic details are lost, making it hard to find a house that truly *feels* right.

### Key Features
Wisteria translates its immense backend complexity into a suite of highly intuitive tools that de-risk and personalise the home-buying process.

*   **The Wisteria Concierge:** A persistent AI strategist (powered by LangChain/LlamaIndex/CrewAI) that manages multi-modal searches (`"Find me something like this image, but in Newcastle"`), conducts proactive trade-off analysis, and pre-compiles search briefs for human agents.
*   **Explainable Valuations (XAI):** Leveraging SHAP, Wisteria provides a narrative waterfall chart of a home's value, visually explaining how specific features (`+£22k for school catchment`, `-£8k for busy road`) dictate the final price.
*   **Comparative Intelligence:** An automated Pro/Con generator and "Persona-Based Scorecard" that conversationally compares favourited properties, eliminating the need for user spreadsheets.
*   **FutureSight Property Report:** An institutional-grade due diligence report projecting future financial scenarios, including a Neighbourhood Evolution Score and a Risk Radar (flood risk, noise pollution).
*   **Dynamic Desire Maps:** Interactive choropleth maps (via Mapbox GL JS) that allow users to visually prospect neighbourhoods based on their own weighted desires (crime rates, pub accessibility, deprivation indices).

---

### Methodology: A Deep Dive
Wisteria's backend is a sophisticated, multi-stage ingestion and inference engine built to handle high-dimensional, interconnected data.

#### 1. Data Ingestion & Feature Engineering
*   **Asynchronous Scraping:** A Mailgun inbound webhook triggers concurrent Apify actors upon receiving new listing emails. The pipeline scrapes raw data/images from portals, executes OCR on EPC registers, and cross-references with third-party Automated Valuation Models (AVMs).
*   **Geospatial Encoding:** Global geographic subsets are transformed into specialised local encodings (`Atlas`, `Compass`, `Microscope`) to produce geographically weighted features and capture complex spatial relationships.
*   **Dimensionality Reduction:** Autoencoders (AE) and Principal Component Analysis (PCA) are utilised for aggressive dimensionality reduction on highly collinear, sparse subsets.

#### 2. The Qualitative Engine: Seeing the Property
The system "sees" and understands properties via an asynchronous, concurrent Gemini API pipeline.
*   **Floorplan Deconstruction:** Gemini analyses floorplans to extract room dimensions, dynamically assigning property images to their respective rooms.
*   **Semantic Tagging:** Images are processed to extract niche qualitative tags (`has_kitchen_island`, `modern_aesthetic`, `renovation_potential`). A dual-head model translates these qualitative sentiments into dense quantitative tensors for the valuation model.
*   **Persona-Based Ratings:** The pipeline evaluates rooms based on simulated demographic "personas" (e.g., *Young Family*, *Professional Couple*), quantifying what features matter most to specific buyer types.

#### 3. The Valuation Model: A "Council of Experts"
A stacked generalisation architecture was developed to eliminate the generalisation gap between training and holdout sets.
*   **Stage 1 (The Specialists):** Multiple specialised LightGBM models are trained on designated feature heads (e.g., census data, spatial data, qualitative tensors) to produce Out-of-Fold (OOF) predictions.
*   **Stage 2 (The Fusion Model):** A final, rigorously regularised LightGBM model is trained on the raw features *plus* the OOF predictions from Stage 1.

---

### Data Sources
The model is built on a rich, multi-modal dataset fused from a wide array of official UK sources:
*   **Property Listings & AVMs:** Scraped data from major UK portals, enhanced with data from `Homipi`, `Bricks&Logic`, and `Mouseprice`.
*   **Environmental & Geospatial:** `Ordnance Survey (OS) Data Hub`, `Copernicus Land Monitoring Service`.
*   **Socio-Economic & Demographic:** `Office for National Statistics (ONS)` (including Census 2021, Output Area Classification), `Consumer Data Research Centre (CDRC)` (including Access to Healthy Assets and Hazards - AHAH).
*   **Government & Administrative:** `Ministry of Housing, Communities & Local Government` (Indices of Deprivation), Energy Performance Certificate (EPC) Registers.

---

### Performance & Limitations
The performance of the valuation model is highly dependent on the quality and volume of historical and spatial training data available.
*   **Robust Accuracy:** The stacked generalisation model achieves a Holdout Mean Absolute Error (MAE) of **£11,448.84** across 10,000 properties in Scotland, England, and Wales. This represents an accuracy of approximately 97%.
*   **Qualitative Blind Spots:** The model currently relies heavily on quantitative data and visual image tagging. It is "naive" to deep, hyper-local qualitative factors—such as hidden structural faults, restrictive covenants, or highly specific local planning disputes—that can significantly alter a property's ceiling price. 

---

### Plans for the future

Using qualitative data: This requires deploying an automated crawler that traverses different Local Planning Authority (LPA) websites to figure out how their website layout works, and how to input a known property address into their search program to download relevant planning histories and documents. 

All PII (Personally Identifiable Information) must be removed!

Documents will need to be tokenised into a tabular format: there will likely be hundreds to thousands of unique tokens, each one signifying whether a certain phrase was mentioned, such as: "the proposed extension is within a conservation area and will require a heritage assessment". This specific constraint would drastically increase renovation costs and restrict the property's potential value uplift. Tokens like these will massively boost performance, as the model will know what different LPAs look for and what hidden restrictions exist on specific streets.

The actual planning permissions submitted by previous owners can be analysed to determine what local authorities historically reject or accept. On top of this, using the quantitative tokens previously mentioned, we can create a predictive program that estimates the likelihood of a buyer successfully gaining planning permission for an extension *before* they buy the house.

Community notes and local neighbourhood forums can be analysed too (after PII has been removed). There may be specific areas with a high rate of noise complaints or active local improvement groups. The final property valuation and FutureSight report can take these community dynamics into account. For future listings, if a
