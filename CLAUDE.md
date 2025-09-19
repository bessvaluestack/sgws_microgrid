# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for the SGWS project feasibility study for a **off grid** basic 12 port EV charging and solar + storage project. It has the following components:
1. Load profile generator for the following cases:
- high load scenario - 10 kW per port observing 14 kW max per pedestal charger 8:30 AM - 5:30 PM every business day
- medium load scenario - employee charging scenario where each car is being charged 20 - 70 kWh with 0.25 - 1 hr mean time between charges (0.66 hr mean) in the 8:30 - 5:30 window
- low load scenario - employee charging scenario where each car is being charged 15 - 50 kWh with 0.5 - 2 hr  time between charges (1.5 hr mean) in the 8:30 - 5:30 window
2. Storage operations model based on solar production (AC) - no grid 
3. Calculation of energy unavailability time and duration 

## Architecture

*Note: Project architecture will be documented here as the codebase develops.*

## Project Structure

## Notes

- Project uses MIT License
