# Project: The Living World Engine

## Overview

This project aims to create a complex, persistent, and dynamic world simulation to serve as the backend for a fantasy game. The world is not a static backdrop but a core character that evolves based on physical laws and is influenced by player actions. The simulation will model geophysics (terrain, erosion), hydrodynamics (water flow), atmospheric science (weather), and sociology (NPC and settlement reactions to environmental changes).

## Core Philosophy

The engine's primary purpose is to be an **Opportunity Engine**. It will create emergent scenarios and problems within the world (droughts, floods, resource shortages) without pre-scripted events. Players have complete freedom to choose their role: they can be heroes who solve these problems, opportunists who profit from them, or catalysts who create new ones.

## Agent Roster & Hierarchy

- **ProjectManager Agent (Lead):** The central coordinator. Delegates tasks, integrates code, and manages the project plan.
- **PhysicsEngine Agent (Specialist):** Implements the core physical laws (water, erosion, weather).
- **LivingWorldAI Agent (Specialist):** Implements the AI for NPCs and settlements.

## Technical Specifications

- **Core Data Structures:**
  - `terrain`: 3D NumPy array (`int8`) where 0=Air, 1=Rock.
  - `water`: 3D NumPy array (`float32`) representing water volume.
- **Code Style:** PEP 8 compliant, clear docstrings, vectorized NumPy operations preferred.

## Project Roadmap & Phases

- **Phase 1 (Complete):** Foundational prototype with basic terrain and water physics.
- **Phase 2 (Next):** Physics Engine Enhancement (Refine Water Physics, Erosion, Weather, Precipitation).
- **Phase 3:** AI Layer Implementation (NPC Awareness, Needs-Based Behavior, Settlement Resources).
- **Phase 4:** Integration & Emergence (Dynamic Systems, Full Loop Test Cases).
- **Phase 5:** Optimization & Scalability.
