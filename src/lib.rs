//! SimCore: the shared library behind ParticleSim (sandbox) and NuclearSim
//! (Nuclear Game). A position-based 2D particle solver with long-range
//! forces, interaction rules and walls; nuclear physics (nuclide data,
//! reactions, point nuclei, reactor parts, heat); the renderer; and the
//! immediate-mode panel GUI and magnifier both apps use.
pub mod rng;
pub mod grid;
pub mod sim;
pub mod types;
pub mod bh;
pub mod nuclear;
pub mod nucdata;
pub mod reactions;
pub mod reactor;
pub mod thermal;
pub mod walls;
pub mod render;
pub mod skin;
pub mod gui;
pub mod zoom;
pub mod game_scenes;
