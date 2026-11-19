# SPDX-License-Identifier: Apache-2.0
from .backbones.contracts import BackboneError

def archive_page_policy(settings, backbone):
    if any((k.startswith(('campaign_', 'a2_experiment_')) for k in settings)):
        raise BackboneError('BLOCKED_CONFIGURATION', 'Only canonical Full archive policy is supported')
    return 'base4k'
