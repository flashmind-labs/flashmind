//! Text and image to 3D generation through Meshy and Tripo.

mod client;

pub use client::{
    AnimationTarget, MeshyTaskOutput, Motion, MotionMode, MotionRequest, RigInput, ThreeDClient,
    ThreeDImage, ThreeDInput, ThreeDOutput, ThreeDProvider, ThreeDRequest,
};
