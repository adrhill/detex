# asdex

## Version `v0.1.1`
* ![Feature][badge-feature] Add `cumsum` primitive handler ([#76])
* ![Feature][badge-feature] Add `erf` to unary elementwise dispatch ([#75])
* ![Bugfix][badge-bugfix] Fix `dot_general` handler for scalar operands ([#75])
* ![Bugfix][badge-bugfix] Fix `gather` handler for wrong ndim in single-dim path ([#75])
* ![Bugfix][badge-bugfix] Handle `batch_group_count > 1` in conv handler ([#73])
* ![Enhancement][badge-enhancement] Factor out primal computation in `fwd_over_rev` and `rev_over_rev` ([#72])
* ![Maintenance][badge-maintenance] Suppress expected warnings for clean pytest output ([#74])
* ![Documentation][badge-docs] Update for PyPI release ([#71])

## Version `v0.1.0`
* ![Feature][badge-feature] Initial release ([#70])


[#76]: https://github.com/adrhill/asdex/pull/76
[#75]: https://github.com/adrhill/asdex/pull/75
[#74]: https://github.com/adrhill/asdex/pull/74
[#73]: https://github.com/adrhill/asdex/pull/73
[#72]: https://github.com/adrhill/asdex/pull/72
[#71]: https://github.com/adrhill/asdex/pull/71
[#70]: https://github.com/adrhill/asdex/pull/70

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg
