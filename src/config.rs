#[derive(serde::Deserialize)]
pub enum FrontFace {
    Cw,
    Ccw,
}
impl Default for FrontFace {
    fn default() -> Self {
        Self::Ccw
    }
}

fn default_vec() -> mint::Vector3<f32> {
    [0.0; 3].into()
}
fn default_scale() -> f32 {
    1.0
}

#[derive(serde::Deserialize)]
pub struct Visual {
    pub model: String,
    #[serde(default)]
    pub front_face: FrontFace,
    #[serde(default = "default_vec")]
    pub pos: mint::Vector3<f32>,
    #[serde(default = "default_vec")]
    pub rot: mint::Vector3<f32>,
    #[serde(default = "default_scale")]
    pub scale: f32,
}
impl Default for Visual {
    fn default() -> Self {
        Self {
            model: String::new(),
            front_face: FrontFace::default(),
            pos: default_vec(),
            rot: default_vec(),
            scale: default_scale(),
        }
    }
}

#[derive(serde::Deserialize)]
pub enum Shape {
    Ball {
        radius: f32,
    },
    Cylinder {
        half_height: f32,
        radius: f32,
    },
    Cuboid {
        half: mint::Vector3<f32>,
    },
    ConvexHull {
        points: Vec<mint::Vector3<f32>>,
        #[serde(default)]
        border_radius: f32,
    },
    TriMesh {
        model: String,
        #[serde(default)]
        convex: bool,
        #[serde(default)]
        border_radius: f32,
    },
}

fn default_friction() -> f32 {
    1.0
}
fn default_restitution() -> f32 {
    0.0
}

#[derive(serde::Deserialize)]
pub struct Collider {
    pub density: f32,
    pub shape: Shape,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default = "default_restitution")]
    pub restitution: f32,
    #[serde(default = "default_vec")]
    pub pos: mint::Vector3<f32>,
    #[serde(default = "default_vec")]
    pub rot: mint::Vector3<f32>,
}

#[derive(serde::Deserialize)]
pub struct AdditionalMass {
    pub density: f32,
    pub shape: Shape,
}

#[derive(serde::Deserialize)]
pub struct Object {
    pub name: String,
    pub visuals: Vec<Visual>,
    pub colliders: Vec<Collider>,
    #[serde(default)]
    pub additional_mass: Option<AdditionalMass>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize)]
pub struct Motor {
    pub stiffness: f32,
    pub damping: f32,
    pub max_force: f32,
}

fn default_time_step() -> f32 {
    0.01
}

#[derive(serde::Deserialize)]
pub struct Engine {
    pub shader_path: String,
    pub data_path: String,
    #[serde(default = "default_time_step")]
    pub time_step: f32,
}
