"""Measured native-attention recipes, independent of retrieval semantics.

SM90 H100/H200 share the measured H200 recipe. SM120 uses the early RTX5090
G4/S1 recipe, whose prep tasks were submitted as a wave, not interleaved units.
Group sizes must divide LOCAL heads: adapting the recipe never pads heads or
changes the selected tokens, budget, precision, or logical Chrono positions.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    version: int
    group: int
    splits: int
    submission: str
    native_splits: int = 1
    reverse: bool = True


HOPPER = DeviceProfile('hopper_h200_g12_s12',3,12,12,'unit')
RTX5090 = DeviceProfile('rtx5090_g4_s1',2,4,1,'wave')
RTX5090_SP8 = DeviceProfile('rtx5090_sp8_local_g_s2',2,5,2,'wave')


def profile_for_capability(capability):
    if tuple(capability)==(9,0):return HOPPER
    if tuple(capability)==(12,0):return RTX5090
    return None


def profile_for_execution(capability, *, physical_sp=1, device_name=None):
    """The SP8 recipe requires the actual 5090 model, not SM120 alone."""
    import re
    base = profile_for_capability(capability)
    if (base is RTX5090 and type(physical_sp) is int and physical_sp == 8
            and isinstance(device_name, str)
            and re.search(r'\bRTX\s*5090\b', device_name, re.IGNORECASE)):
        return RTX5090_SP8
    return base


def device_profile(device, *, physical_sp=1):
    import torch
    if device is None or torch.device(device).type!='cuda':return None
    capability = torch.cuda.get_device_capability(device)
    # Leave the existing device/SP paths and their discovery calls intact.
    name = torch.cuda.get_device_name(device) if physical_sp == 8 and tuple(capability) == (12,0) else None
    return profile_for_execution(capability, physical_sp=physical_sp, device_name=name)


def legal_group(heads, preferred):
    if type(heads) is not int or heads<1 or type(preferred) is not int or preferred<1:
        raise ValueError('positive local heads and preferred head group required')
    return next(g for g in range(min(heads,preferred),0,-1) if heads%g==0)


def transport_group(config, parameters, heads):
    group=(config or {}).get('head_group')
    if group is None:
        group=heads if parameters.get('consumer','original')=='original' else int(parameters['head_group'])
    if type(group) is not int or group<1 or heads%group:
        raise ValueError('native head group must divide actual local heads')
    return group


def native_version_for_plan(plan, device):
    # Installing FA3 alone must not change other algorithms or baselines.
    # DraftKV's explicit device policy, however, takes precedence over an old
    # generic native_attention_version=2 field on H100/H200.
    profile=device_profile(device) if plan.get('algorithm')=='draftkv' else None
    return profile.version if profile else int(plan.get('native_attention_version',2))
