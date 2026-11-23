//! Automation: a controller that holds a reading at a set point by moving
//! the control rods (fission) or throttling the spawners (beams and fuel
//! feeds: fusion or fission), and rules that act when something happens
//! (at a time, every so often, when a reading crosses a level, at the first
//! reaction of a kind): trips that SCRAM, timed rod moves, beam pulses.

use crate::nuclear::EventKind;
use crate::reactor::{Reactor, SAMPLE_DT};
use crate::sim::Sim;

/// A reading the controller and the rules can watch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Measure {
    FissionRate,
    FusionRate,
    /// Net reaction power (W): released minus absorbed.
    ReactionPower,
    /// Decay heat (W).
    DecayHeat,
    Neutrons,
    Temperature,
    /// Reactor period (s): how fast the neutron population grows (only
    /// positive, rising periods count; steady or falling reads as infinite).
    Period,
    /// Count rate of detector k (per s).
    Detector(usize),
}

impl Measure {
    pub const ALL: [Measure; 8] = [
        Measure::FissionRate,
        Measure::FusionRate,
        Measure::ReactionPower,
        Measure::DecayHeat,
        Measure::Neutrons,
        Measure::Temperature,
        Measure::Period,
        Measure::Detector(0),
    ];

    pub fn name(self) -> String {
        match self {
            Measure::FissionRate => "fission rate".into(),
            Measure::FusionRate => "fusion rate".into(),
            Measure::ReactionPower => "net reaction power".into(),
            Measure::DecayHeat => "decay heat".into(),
            Measure::Neutrons => "free neutrons".into(),
            Measure::Temperature => "temperature".into(),
            Measure::Period => "reactor period".into(),
            Measure::Detector(k) => format!("detector {} rate", k + 1),
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Measure::FissionRate => "Fission",
            Measure::FusionRate => "Fusion",
            Measure::ReactionPower => "Power",
            Measure::DecayHeat => "Decay heat",
            Measure::Neutrons => "Neutrons",
            Measure::Temperature => "Temp.",
            Measure::Period => "Period",
            Measure::Detector(_) => "Detector",
        }
    }

    /// Format a value of this reading.
    pub fn format(self, v: f64) -> String {
        match self {
            Measure::FissionRate | Measure::FusionRate | Measure::Detector(_) => format!("{} /s", sig(v)),
            Measure::ReactionPower | Measure::DecayHeat => watts(v),
            Measure::Neutrons => format!("{v:.0}"),
            Measure::Temperature => crate::thermal::format_kelvin(v),
            Measure::Period => {
                if v.is_finite() { format!("{v:.0} s") } else { "∞".into() }
            }
        }
    }

    /// Sensible range for a set point or level slider.
    pub fn range(self) -> (f64, f64) {
        match self {
            Measure::FissionRate | Measure::FusionRate | Measure::Detector(_) => (0.01, 1000.0),
            Measure::ReactionPower | Measure::DecayHeat => (1e-15, 1e-6),
            Measure::Neutrons => (1.0, 5000.0),
            Measure::Temperature => (300.0, 1e9),
            Measure::Period => (1.0, 300.0),
        }
    }

    fn code(self) -> String {
        match self {
            Measure::FissionRate => "fission".into(),
            Measure::FusionRate => "fusion".into(),
            Measure::ReactionPower => "power".into(),
            Measure::DecayHeat => "decayheat".into(),
            Measure::Neutrons => "neutrons".into(),
            Measure::Temperature => "temp".into(),
            Measure::Period => "period".into(),
            Measure::Detector(k) => format!("det{k}"),
        }
    }

    fn from_code(s: &str) -> Option<Self> {
        Some(match s {
            "fission" => Measure::FissionRate,
            "fusion" => Measure::FusionRate,
            "power" => Measure::ReactionPower,
            "decayheat" => Measure::DecayHeat,
            "neutrons" => Measure::Neutrons,
            "temp" => Measure::Temperature,
            "period" => Measure::Period,
            _ => Measure::Detector(s.strip_prefix("det")?.parse().ok()?),
        })
    }

    /// Current value, averaged over the last second (None before any sample).
    pub fn value(self, sim: &Sim) -> Option<f64> {
        self.average(sim, 4)
    }

    /// Value averaged over the last `k` samples.
    pub fn average(self, sim: &Sim, k: usize) -> Option<f64> {
        let re = &sim.reactor;
        if let Measure::Detector(k) = self {
            return re.detectors.get(k).map(|d| d.rate as f64);
        }
        if let Measure::Period = self {
            return Some(match re.period() {
                Some(p) if p.is_finite() && p > 0.0 => p as f64,
                _ => f64::INFINITY,
            });
        }
        let k = re.samples.len().min(k.max(1));
        if k == 0 {
            return None;
        }
        let f = |s: &crate::reactor::Sample| -> f64 {
            match self {
                Measure::FissionRate => s.fission_rate as f64,
                Measure::FusionRate => s.fusion_rate as f64,
                Measure::ReactionPower => s.net_w(),
                Measure::DecayHeat => s.decay_w,
                Measure::Neutrons => s.neutrons as f64,
                Measure::Temperature => s.temperature_k,
                _ => 0.0,
            }
        };
        Some(re.samples.iter().rev().take(k).map(f).sum::<f64>() / k as f64)
    }
}

fn sig(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// "3.2 pW".
pub fn watts(w: f64) -> String {
    if w == 0.0 {
        return "0 W".into();
    }
    let units = [(1e-18, "aW"), (1e-15, "fW"), (1e-12, "pW"), (1e-9, "nW"), (1e-6, "µW"), (1e-3, "mW"), (1.0, "W")];
    let a = w.abs();
    let (k, u) = units.iter().rev().find(|(k, _)| a >= *k).copied().unwrap_or(units[0]);
    let x = w / k;
    let digits = if x.abs() >= 100.0 { 0 } else if x.abs() >= 10.0 { 1 } else { 2 };
    format!("{x:.digits$} {u}")
}

/// What the controller moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Actuator {
    /// The control-rod bank (fission).
    Rods,
    /// The rate of every spawner: beams and fuel feeds (fusion, or a
    /// molten-salt fuel feed).
    Feeds,
}

/// Holds `measure` at `setpoint`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controller {
    pub on: bool,
    pub measure: Measure,
    pub setpoint: f64,
    pub actuator: Actuator,
    /// How hard it pushes (1 = default).
    pub gain: f32,
    last_t: f64,
}

impl Default for Controller {
    fn default() -> Self {
        Self { on: false, measure: Measure::FissionRate, setpoint: 2.0, actuator: Actuator::Rods, gain: 1.0, last_t: -1.0 }
    }
}

/// Logarithmic error, clamped: positive when the reading is too high.
fn log_error(m: f64, set: f64) -> f64 {
    let floor = set.abs() * 0.05 + 1e-30;
    ((m.max(0.0) + floor) / (set.abs() + floor)).ln().clamp(-2.0, 2.0)
}

/// Growth rate (per s) of the reading over the last ~6 s.
fn growth(sim: &Sim, m: Measure) -> f64 {
    let s = &sim.reactor.samples;
    if s.len() < 24 {
        return 0.0;
    }
    let val = |x: &crate::reactor::Sample| -> f64 {
        match m {
            Measure::FissionRate => x.fission_rate as f64,
            Measure::FusionRate => x.fusion_rate as f64,
            Measure::ReactionPower => x.net_w(),
            Measure::DecayHeat => x.decay_w,
            Measure::Neutrons => x.neutrons as f64,
            Measure::Temperature => x.temperature_k,
            _ => 0.0,
        }
    };
    let v: Vec<f64> = s.iter().rev().take(24).map(val).collect();
    let newer = v[..8].iter().sum::<f64>() / 8.0;
    let older = v[16..].iter().sum::<f64>() / 8.0;
    let floor = newer.abs().max(older.abs()) * 0.1 + 1e-30;
    (((newer.max(0.0) + floor) / (older.max(0.0) + floor)).ln() / (16.0 * SAMPLE_DT as f64)).clamp(-1.0, 1.0)
}

impl Controller {
    /// A controller, switched on.
    pub fn new(measure: Measure, setpoint: f64, actuator: Actuator) -> Self {
        Self { on: true, measure, setpoint, actuator, ..Self::default() }
    }

    fn update(&mut self, sim: &mut Sim) {
        if !self.on {
            return;
        }
        // Act once per new sample.
        let Some(last) = sim.reactor.samples.back() else { return };
        if last.t <= self.last_t {
            return;
        }
        self.last_t = last.t;
        // A few seconds of readings: reaction rates are counts, and noisy.
        let Some(m) = self.measure.average(sim, 16) else { return };
        // A period reading is held *above* its set point: shorter = too fast.
        let (err, rate) = if self.measure == Measure::Period {
            let p = if m.is_finite() { m } else { 1e6 };
            (-(p / self.setpoint.max(1e-3)).ln().clamp(-2.0, 2.0), 0.0)
        } else {
            (log_error(m, self.setpoint), growth(sim, self.measure))
        };
        // Proportional on the (log) error plus a strong damping on the
        // growth rate: delayed neutrons make a reactor slow to answer, so
        // the controller watches where it is heading, not just where it is.
        let u = self.gain as f64 * (0.5 * err + 3.0 * rate);
        let re = &mut sim.reactor;
        match self.actuator {
            Actuator::Rods => {
                if re.rods.is_empty() || re.scram {
                    return;
                }
                re.target = (re.target + (u * 0.006) as f32).clamp(0.0, 1.0);
            }
            Actuator::Feeds => {
                re.throttle = (re.throttle - (u * 0.02) as f32).clamp(0.0, 1.0);
            }
        }
    }

    pub fn to_line(&self) -> String {
        format!(
            "controller {} {} {} {} {}",
            self.on as u8,
            self.measure.code(),
            self.setpoint,
            match self.actuator {
                Actuator::Rods => "rods",
                Actuator::Feeds => "feeds",
            },
            self.gain
        )
    }

    pub fn from_fields(f: &[&str]) -> Option<Self> {
        Some(Self {
            on: f.get(1)? == &"1",
            measure: Measure::from_code(f.get(2)?)?,
            setpoint: f.get(3)?.parse().ok()?,
            actuator: if f.get(4)? == &"feeds" { Actuator::Feeds } else { Actuator::Rods },
            gain: f.get(5).and_then(|g| g.parse().ok()).unwrap_or(1.0),
            last_t: -1.0,
        })
    }
}

/// When a rule fires.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Trigger {
    /// Once, at sim time t (s).
    At(f64),
    /// Every so many seconds (first after one period).
    Every(f64),
    /// When the reading rises above the level (again after it has dropped
    /// back below).
    Above(Measure, f64),
    /// When it falls below the level.
    Below(Measure, f64),
    /// At the first reaction of this kind after the rule was made.
    First(EventKind),
}

impl Trigger {
    pub fn describe(self) -> String {
        match self {
            Trigger::At(t) => format!("at t = {t:.1} s"),
            Trigger::Every(t) => format!("every {t:.1} s"),
            Trigger::Above(m, v) => format!("when {} > {}", m.name(), m.format(v)),
            Trigger::Below(m, v) => format!("when {} < {}", m.name(), m.format(v)),
            Trigger::First(k) => format!("at the first {}", k.name()),
        }
    }
}

/// What a rule does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    /// Drop every rod in.
    Scram,
    /// Drive the rod bank to this insertion (0 = out, 1 = in).
    Rods(f32),
    /// Set the spawner throttle (0 = every beam and feed off, 1 = full).
    Throttle(f32),
    /// Every spawner fires this many extra nuclei at once.
    Burst(u32),
    /// Set the neutron sources' strength (n/s each).
    Sources(f32),
    /// Pause the simulation.
    Pause,
}

impl Action {
    pub fn describe(self) -> String {
        match self {
            Action::Scram => "SCRAM".into(),
            Action::Rods(f) => format!("rods to {:.0}% in", f * 100.0),
            Action::Throttle(0.0) => "stop the beams and feeds".into(),
            Action::Throttle(f) => format!("beams and feeds to {:.0}%", f * 100.0),
            Action::Burst(k) => format!("each spawner fires {k}"),
            Action::Sources(r) => format!("neutron sources to {r:.2} n/s"),
            Action::Pause => "pause".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    pub trigger: Trigger,
    pub action: Action,
    pub enabled: bool,
    /// Times it has fired.
    pub fired: u32,
    /// Level rules: the condition was true last check (fires on the edge).
    was: bool,
    /// Every: next firing time. First: the count it started from (-1 = not set).
    next: f64,
}

impl Rule {
    pub fn new(trigger: Trigger, action: Action) -> Self {
        Self { trigger, action, enabled: true, fired: 0, was: false, next: -1.0 }
    }

    pub fn describe(&self) -> String {
        format!("{} → {}", self.trigger.describe(), self.action.describe())
    }

    /// Start over (fires again from now).
    pub fn rearm(&mut self) {
        self.fired = 0;
        self.was = false;
        self.next = -1.0;
    }

    pub fn to_line(&self) -> String {
        let t = match self.trigger {
            Trigger::At(t) => format!("at {t}"),
            Trigger::Every(t) => format!("every {t}"),
            Trigger::Above(m, v) => format!("above {} {v}", m.code()),
            Trigger::Below(m, v) => format!("below {} {v}", m.code()),
            Trigger::First(k) => format!("first {}", k.name()),
        };
        let a = match self.action {
            Action::Scram => "scram 0".to_string(),
            Action::Rods(f) => format!("rods {f}"),
            Action::Throttle(f) => format!("throttle {f}"),
            Action::Burst(k) => format!("burst {k}"),
            Action::Sources(r) => format!("sources {r}"),
            Action::Pause => "pause 0".to_string(),
        };
        format!("rule {} {t} {a}", self.enabled as u8)
    }

    pub fn from_fields(f: &[&str]) -> Option<Self> {
        let enabled = f.get(1)? == &"1";
        let (trigger, k) = match *f.get(2)? {
            "at" => (Trigger::At(f.get(3)?.parse().ok()?), 4),
            "every" => (Trigger::Every(f.get(3)?.parse().ok()?), 4),
            "above" => (Trigger::Above(Measure::from_code(f.get(3)?)?, f.get(4)?.parse().ok()?), 5),
            "below" => (Trigger::Below(Measure::from_code(f.get(3)?)?, f.get(4)?.parse().ok()?), 5),
            "first" => {
                let name = *f.get(3)?;
                (Trigger::First(*EventKind::ALL.iter().find(|e| e.name() == name)?), 4)
            }
            _ => return None,
        };
        let x = f.get(k + 1).copied().unwrap_or("0");
        let action = match *f.get(k)? {
            "scram" => Action::Scram,
            "rods" => Action::Rods(x.parse().ok()?),
            "throttle" => Action::Throttle(x.parse().ok()?),
            "burst" => Action::Burst(x.parse().ok()?),
            "sources" => Action::Sources(x.parse().ok()?),
            "pause" => Action::Pause,
            _ => return None,
        };
        Some(Self { enabled, ..Self::new(trigger, action) })
    }
}

/// Running count of reactions of a kind.
fn kind_count(sim: &Sim, k: EventKind) -> u64 {
    let st = &sim.nuclear;
    match k {
        EventKind::Fission => st.fissions,
        EventKind::Fusion => st.fusions,
        EventKind::Breakup => st.breakups,
        EventKind::Capture => st.captures,
        EventKind::Decay => st.counts.values().sum(),
        EventKind::Annihilation => st.annihilations,
        _ => st.dropped + st.history.len() as u64,
    }
}

fn fire(re: &mut Reactor, a: Action) {
    match a {
        Action::Scram => re.scram(),
        Action::Rods(f) => {
            re.target = f.clamp(0.0, 1.0);
            re.scram = false;
        }
        Action::Throttle(f) => re.throttle = f.clamp(0.0, 1.0),
        Action::Burst(k) => re.spawners.iter_mut().for_each(|s| s.burst += k),
        Action::Sources(r) => re.source_rate = r.max(0.0),
        Action::Pause => re.pause_request = true,
    }
}

/// Run the controller and the rules. Call once per frame after sampling.
pub fn update(sim: &mut Sim) {
    let mut c = sim.reactor.controller;
    c.update(sim);
    sim.reactor.controller = c;
    if sim.reactor.rules.is_empty() {
        return;
    }
    let now = sim.nuclear.time;
    let mut rules = std::mem::take(&mut sim.reactor.rules);
    let mut fired: Vec<String> = Vec::new();
    for r in rules.iter_mut() {
        if !r.enabled {
            continue;
        }
        let go = match r.trigger {
            Trigger::At(t) => r.fired == 0 && now >= t,
            Trigger::Every(p) => {
                if r.next < 0.0 {
                    r.next = now + p.max(0.05);
                }
                if now >= r.next {
                    r.next += p.max(0.05);
                    if r.next < now {
                        r.next = now + p.max(0.05);
                    }
                    true
                } else {
                    false
                }
            }
            Trigger::Above(m, v) | Trigger::Below(m, v) => {
                let above = matches!(r.trigger, Trigger::Above(..));
                let cond = m.value(sim).is_some_and(|x| if above { x > v } else { x < v && sim.reactor.samples.len() >= 8 });
                let edge = cond && !r.was;
                r.was = cond;
                edge
            }
            Trigger::First(k) => {
                let c = kind_count(sim, k) as f64;
                if r.next < 0.0 {
                    r.next = c;
                }
                r.fired == 0 && c > r.next
            }
        };
        if go {
            r.fired += 1;
            fire(&mut sim.reactor, r.action);
            fired.push(r.describe());
        }
    }
    sim.reactor.rules = rules;
    for d in fired {
        sim.nuclear.record(EventKind::Other, format!("rule: {d}  (automation, at t = {now:.1} s)"));
        sim.reactor.notices.push(format!("Rule: {d}"));
    }
}
