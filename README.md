# Wisteria: An Intelligence-Driven Real Estate & PropTech Platform
[![Status](https://img.shields.io/badge/status-in%20development-green)](https://github.com/brandonhenson/wisteria)

**Wisteria is a bleeding-edge real estate listings portal designed to transcend the static data tables of traditional platforms like Rightmove or Zoopla. By fusing multi-modal property data with sophisticated AI, Wisteria provides automated valuations, deep qualitative insights, and a truly personalized user experience, aiming to eliminate buyer "analysis paralysis" and build institutional-grade trust.**

---

### Table of Contents
1.  [The Problem with PropTech Today](#the-problem-with-proptech-today)
2.  [The Solution: Wisteria's User-Centric Features](#the-solution-wisterias-user-centric-features)
3.  [Architectural Deep Dive](#architectural-deep-dive)
    *   [Data Ingestion & Feature Engineering](#1-data-ingestion--feature-engineering)
    *   [The Qualitative Engine: Seeing the Property](#2-the-qualitative-engine-seeing-the-property)
    *   [The Valuation Model: A "Council of Experts"](#3-the-valuation-model-a-council-of-experts)
4.  [Tech Stack](#tech-stack)
5.  [Data Sources](#data-sources)

---

### The Problem with PropTech Today
Traditional real estate portals present data as disconnected, static tables. This approach burdens the user with immense cognitive load, leading to:
*   **Analysis Paralysis:** Users are forced to manually compare dozens of listings across spreadsheets.
*   **Opaque Valuations:** Simple price estimates lack transparency, eroding user trust.
*   **Qualitative Blind Spots:** Key lifestyle and aesthetic details are lost, making it hard to find a house that truly *feels* right.

### The Solution: Wisteria's User-Centric Features

Wisteria translates its immense backend complexity into a suite of highly intuitive tools that de-risk and personalize the home-buying process.

*   💬 **The Wisteria Concierge:** A persistent AI strategist (LangChain/LlamaIndex/CrewAI) that manages multi-modal searches (`"Find me something like this image, but in Newcastle"`), conducts proactive trade-off analysis, and pre-compiles search briefs for human agents.
*   💡 **Explainable Valuations (XAI):** Leveraging SHAP, Wisteria provides a narrative waterfall chart of a home's value, visually explaining how specific features (`+£22k for school catchment`, `-£8k for busy road`) dictate the final price.
*   ⚖️ **Comparative Intelligence:** An automated Pro/Con generator and "Persona-Based Scorecard" that conversationally compares favorited properties, eliminating the need for user spreadsheets.
*   📈 **FutureSight Property Report:** An institutional-grade due diligence report projecting future financial scenarios, including a Neighborhood Evolution Score and a Risk Radar (flood risk, noise pollution).
*   🗺️ **Dynamic Desire Maps:** Interactive choropleth maps (Mapbox GL JS) that allow users to visually prospect neighborhoods based on their own weighted desires (crime rates, pub accessibility, deprivation indices).

---

### Architectural Deep Dive
Wisteria's backend is a sophisticated, multi-stage ingestion and inference engine built to handle high-dimensional, interconnected data.

#### 1. Data Ingestion & Feature Engineering
*   **Asynchronous Scraping:** A Mailgun inbound webhook triggers concurrent Apify actors upon receiving new listing emails. The pipeline scrapes raw data/images from portals, executes OCR on EPC registers, and cross-references with third-party Automated Valuation Models (AVMs).
*   **Geospatial Encoding:** Global geographic subsets are transformed into specialized local encodings (`Atlas`, `Compass`, `Microscope`) to produce geographically weighted features and capture complex spatial relationships.
*   **Dimensionality Reduction:** Autoencoders (AE) and Principal Component Analysis (PCA) are utilized for aggressive dimensionality reduction on highly collinear, sparse subsets.

#### 2. The Qualitative Engine: Seeing the Property
The system "sees" and understands properties via an asynchronous, concurrent Gemini API pipeline.
*   **Floorplan Deconstruction:** Gemini analyzes floorplans to extract room dimensions, dynamically assigning property images to their respective rooms.
*   **Semantic Tagging:** Images are processed to extract niche qualitative tags (`has_kitchen_island`, `modern_aesthetic`, `renovation_potential`). A dual-head model translates these qualitative sentiments into dense quantitative tensors for the valuation model.
*   **Persona-Based Ratings:** The pipeline evaluates rooms based on simulated demographic "personas" (e.g., *Young Family*, *Professional Couple*), quantifying what features matter most to specific buyer types.

#### 3. The Valuation Model: A "Council of Experts"
A stacked generalization architecture was developed to eliminate the generalization gap between training and holdout sets.
*   **Stage 1 (The Specialists):** Multiple specialized LightGBM models are trained on designated feature heads (e.g., census data, spatial data, qualitative tensors) to produce Out-of-Fold (OOF) predictions.
*   **Stage 2 (The Fusion Model):** A final, rigorously regularized LightGBM model is trained on the raw features *plus* the OOF predictions from Stage 1.

> **Result:** A highly robust **Holdout Mean Absolute Error (MAE) of £11,448.84** across 10,000 properties in Scotland, England, and Wales. This represents an accuracy of approximately 97%.

---

### 🛠️ Tech Stack
*   **AI & Machine Learning:** LightGBM, Scikit-Learn, TensorFlow/Keras (for Autoencoders), SHAP (for XAI), LangChain, LlamaIndex, CrewAI, Gemini API.
*   **Backend & Data Processing:** Python, Apify, Mailgun API, Webhooks, Asynchronous Processing.
*   **Frontend & Visualization:** Mapbox GL JS.

---

### 📚 Data Sources
*   **Property Listings & AVMs:** Scraped data from major UK portals, enhanced with data from `Homipi`, `Bricks&Logic`, and `Mouseprice`.
*   **Environmental & Geospatial:** `Ordnance Survey (OS) Data Hub`, `Copernicus Land Monitoring Service`.
*   **Socio-Economic & Demographic:** `Office for National Statistics (ONS)` (including Census 2021, Output Area Classification), `Consumer Data Research Centre (CDRC)` (including Access to Healthy Assets and Hazards - AHAH).
*   **Government & Administrative:** `Ministry of Housing, Communities & Local Government` (Indices of Deprivation), Energy Performance Certificate (EPC) Registers.
