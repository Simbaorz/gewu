# Gewu Core

`gewu-core` contains process-wide infrastructure and stable utility primitives shared by Gewu
applications and packages:

- UUID and UTC time helpers
- byte-size parsing
- two-stage dotenv/bootstrap and typed local or Apollo YAML configuration loading
- synchronous bootstrap logging and bounded asynchronous process logging
- isolated blocking and filesystem execution lanes
- event-loop lag sampling and sustained-capacity readiness evaluation

Application-specific settings remain in the owning application. Libraries may use the pure
utility modules, but only process entrypoints should load configuration or configure logging.

The base install contains only those shared primitives. Apollo, relational database, Redis, and
HTTP infrastructure are selected explicitly with the `apollo`, `database`, `redis`, and `http`
extras; `all` is available only for processes that intentionally use every stack.

`ApolloBootstrapSettings` keeps Config Service identity in the local bootstrap layer and selects
Apollo with `CONFIG_SOURCE=apollo`. `SettingsRuntime` performs the initial ordered Namespace
merge, applies process-environment overrides, validates the application-owned Pydantic model, and
can continuously long-poll for valid changes. It writes only validated last-known-good snapshots
to a private atomic cache. Processes that do not need monitoring may call `load_settings_once`.
