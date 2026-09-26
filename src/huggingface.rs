//! Optional Hugging Face Hub checkpoint retrieval.
//!
//! Enable the `huggingface` Cargo feature. Downloads use `hf-hub`'s standard
//! cache and honor its `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, and
//! `HF_ENDPOINT` environment variables.

use std::path::{Path, PathBuf};

use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};

use crate::{Error, Result, Sam2Variant};

/// Parameters for resolving one published checkpoint through the Hub cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HuggingFaceCheckpoint {
    pub repository: String,
    pub filename: String,
    pub revision: Option<String>,
    pub cache_dir: Option<PathBuf>,
    pub local_files_only: bool,
    pub force_download: bool,
}

impl HuggingFaceCheckpoint {
    /// Uses Meta's published repository and checkpoint filename for `variant`.
    pub fn for_variant(variant: Sam2Variant) -> Self {
        Self {
            repository: variant.hugging_face_model_id().to_owned(),
            filename: variant.checkpoint_file().to_owned(),
            revision: None,
            cache_dir: None,
            local_files_only: false,
            force_download: false,
        }
    }

    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    pub fn cache_dir(mut self, cache_dir: impl AsRef<Path>) -> Self {
        self.cache_dir = Some(cache_dir.as_ref().to_owned());
        self
    }

    pub fn local_files_only(mut self, enabled: bool) -> Self {
        self.local_files_only = enabled;
        self
    }

    pub fn force_download(mut self, enabled: bool) -> Self {
        self.force_download = enabled;
        self
    }

    /// Returns a stable cache path, downloading the file when necessary.
    pub fn download(&self) -> Result<PathBuf> {
        if self.repository.trim().is_empty() || self.filename.trim().is_empty() {
            return Err(Error::Integration {
                operation: "invalid Hugging Face checkpoint request",
                message: "repository and filename must not be empty".into(),
            });
        }
        let revision = self.revision.as_deref().unwrap_or("main").to_owned();
        let repo = Repo::with_revision(self.repository.clone(), RepoType::Model, revision);
        let cache_dir = self.cache_dir.clone().or_else(|| {
            std::env::var_os("HF_HUB_CACHE")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
        });
        if self.local_files_only {
            let cache = cache_dir.map_or_else(Cache::from_env, Cache::new);
            return cache
                .repo(repo)
                .get(&self.filename)
                .ok_or_else(|| Error::Integration {
                    operation: "Hugging Face checkpoint cache lookup failed",
                    message: format!(
                        "{} was not found in the local cache for {}",
                        self.filename, self.repository
                    ),
                });
        }
        let mut builder = ApiBuilder::from_env().with_progress(false);
        if let Some(cache_dir) = cache_dir {
            builder = builder.with_cache_dir(cache_dir);
        }
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.trim().is_empty() {
                builder = builder.with_token(Some(token));
            }
        }
        let api = builder.build().map_err(hub_error)?;
        let repo = api.repo(repo);
        if self.force_download {
            repo.download(&self.filename).map_err(hub_error)
        } else {
            repo.get(&self.filename).map_err(hub_error)
        }
    }
}

/// Downloads the published checkpoint for `variant` to the standard Hub cache.
pub fn download_checkpoint(variant: Sam2Variant) -> Result<PathBuf> {
    HuggingFaceCheckpoint::for_variant(variant).download()
}

fn hub_error(error: impl std::fmt::Display) -> Error {
    Error::Integration {
        operation: "Hugging Face checkpoint download failed",
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_variants_map_to_their_checkpoint_repositories() {
        for variant in Sam2Variant::ALL {
            let request = HuggingFaceCheckpoint::for_variant(variant);
            assert_eq!(request.repository, variant.hugging_face_model_id());
            assert_eq!(request.filename, variant.checkpoint_file());
            assert_eq!(request.revision, None);
            assert!(!request.local_files_only);
            assert!(!request.force_download);
        }
    }

    #[test]
    #[ignore = "requires network access"]
    fn downloads_and_then_resolves_from_the_local_cache() {
        let cache = tempfile::tempdir().unwrap();
        let mut request = HuggingFaceCheckpoint::for_variant(Sam2Variant::HieraTiny)
            .cache_dir(cache.path())
            .force_download(true);
        request.filename = "README.md".into();
        let downloaded = request.download().unwrap();
        assert!(downloaded.metadata().unwrap().len() > 0);

        request.force_download = false;
        request.local_files_only = true;
        assert_eq!(request.download().unwrap(), downloaded);
    }
}
