# SolarSense-AI Dashboard

This is the frontend dashboard for the SolarSense-AI space-weather operational intelligence engine. It visualizes solar flare nowcasts and forecasts using Soft X-ray and Hard X-ray time-series data from ISRO Aditya-L1 instruments (SoLEXS and HEL1OS).

## Key Pages

1. Live Monitoring: Real-time flux line charts, active probabilities, warning alerts, and physics-grounded rationales.
2. Historical Analytics: Query system predictions and check the database validation report.
3. Model Performance: Evaluation metrics (True Skill Statistic, Heidke Skill Score, False Alarm Ratio) for the trained ensemble.
4. Diagnostics: Real-time telemetry, missing value counts, and database statistics.

## Project Structure

* src/components: UI widgets, layout utilities, and chart components.
* src/pages: Dashboard sub-pages.
* src/services: API integration layer.
* src/hooks: Custom React state hooks.

## Prerequisites

* Node.js (version 18 or above recommended)
* npm (package manager)

## Setup and Running

1. Install dependencies:
   npm install

2. Run the local development server:
   npm run dev

3. Build the production application:
   npm run build
