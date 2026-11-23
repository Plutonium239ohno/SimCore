# SimCore

Shared library for [ParticleSim](https://github.com/Plutonium239ohno/ParticleSim)
and [NuclearSim](https://github.com/Plutonium239ohno/NuclearSim).

- `sim` — position-based particle solver (collisions, gravity, Coulomb with
  Barnes–Hut, interaction-rule matrix, nuclear clusters and point nuclei)
- `walls` — wall segments: insulators, conductors, freeze plugs, filters
- `nuclear`, `nucdata`, `reactions`, `reactor`, `thermal`, `game_scenes` —
  nuclear physics and the Nuclear Game's scenarios
- `render`, `gui`, `zoom`, `skin` — drawing, panel GUI, magnifier, font

```
cargo test --release                       # library tests
cargo run --release --example bench        # solver benchmark
cargo run --release --example reactor_probe
```
