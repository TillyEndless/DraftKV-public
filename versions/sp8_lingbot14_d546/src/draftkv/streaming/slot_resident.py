"""Use existing native CUDA ranges when a unit needs no receive-slot storage."""
def direct_resident_unit(resident, group, begin, end, head_group, receive=None):
    # Temporal assignment changes K. Such units retain the established copy
    # and prepare path, as do pinned CPU ranges and mixed history units.
    if any(begin<start+value.shape[1] and end>start for start,value in resident.temporal):
        return None
    for start,key,value in resident.segments:
        if begin>=start and end<=start+key.shape[1] and key.device.type=='cuda' and value.device.type=='cuda':
            if receive is not None and any(t.untyped_storage().data_ptr()==receive.untyped_storage().data_ptr() for t in (key,value)):
                return None
            hs=slice(group*head_group,(group+1)*head_group)
            return (key[:,begin-start:end-start,hs],value[:,begin-start:end-start,hs])
    return None
