/*
 * Copyright 2026 XclipseTools
 * SPDX-License-Identifier: MIT
 *
 * The part of the kbase backend that is NOT a pan_kmod op: job submission.
 *
 * pan_kmod deliberately stops at memory and VMs -- submission is per-driver and lives in the
 * Vulkan driver, which is why panvk has separate jm/ and csf/ queue backends. kbase is a third
 * shape, so it needs a third door, and this is it.
 */

#ifndef KBASE_KMOD_H
#define KBASE_KMOD_H

#include <stdbool.h>
#include <stdint.h>

struct pan_kmod_dev;
struct pan_kmod_bo;

/* core_req values are TAKEN VERBATIM from the vendor driver's own atoms, captured while it
 * rendered a triangle (probe/mali/glprobe2.c, logs/mspy_gl.txt). The frame arrived as ONE
 * JOB_SUBMIT carrying two atoms:
 *
 *    atom 1  jc=...1040  core_req=0x4e    pre_dep = none
 *    atom 2  jc=...1540  core_req=0x8001  pre_dep = {atom 1, type 1}
 *
 * which is exactly the shape panvk already produces -- a vertex/tiler chain and a fragment chain
 * that waits for it. Guessing these from a bit table would have been an invitation to be subtly
 * wrong; the numbers below are what the hardware was actually asked for.
 */
#define KBASE_JD_REQ_VERTEX_TILER 0x4eu
#define KBASE_JD_REQ_FRAGMENT     0x8001u
/* BASE_JD_REQ_PERMON: kbase starts the cycle counter for the atom, which is also what makes the
 * GPU's system timestamp propagate (WRITE_VALUE SYSTEM_TIMESTAMP stores 0 without it). */
#define KBASE_JD_REQ_PERMON       0x80u

/* Dependency type 1, as observed. kbase names 0 invalid, 1 data, 2 order. */
#define KBASE_JD_DEP_DATA 1u

struct pan_kmod_kbase_atom {
   uint64_t jc;           /* GPU VA of the job chain head */
   uint32_t core_req;     /* one of the KBASE_JD_REQ_* above */
   uint8_t atom_number;   /* 1..255; kbase treats 0 as unset */
   uint8_t dep_atom;      /* atom_number this one waits for, or 0 for none */
};

/* True when this device is driven by kbase rather than a DRM node. Every caller that would
 * otherwise reach for a DRM ioctl has to ask, because -- measured, see kbase_kmod.c -- this
 * kernel returns SUCCESS for DRM ioctls it does not implement. */
bool pan_kmod_dev_is_kbase(const struct pan_kmod_dev *dev);

/* Sync objects for kbase, which has none: binary objects kept in this process, with the DRM
 * syncobj semantics Mesa's runtime expects (see kbase_kmod.c). Each call returns a new provider
 * over the same process-wide table; finalize frees the provider, not the objects. */
struct util_sync_provider *pan_kmod_kbase_sync_provider(void);

/* The most atoms one submission takes: atom numbers are 1..255 and unique among the atoms in
 * flight, and a submission may add a fence-trigger atom. */
#define PAN_KMOD_KBASE_MAX_ATOMS 253

/* Submit atoms without waiting for them. atom_number is 1..nr_atoms within the call and dep_atom
 * refers to those (0: none). The atoms run after everything submitted before on the device. When
 * fence_fd is not NULL it receives a sync file (the caller owns it) that signals once they have
 * all run, or -1 when there is nothing to wait for. done, if set, is called on the kmod's event
 * thread when every atom has reported, with ok false if one faulted. Returns the submission's
 * sequence number, or 0 when the kernel refused it (done is not called then). */
typedef void (*pan_kmod_kbase_done_cb)(void *data, bool ok);
uint64_t pan_kmod_kbase_submit_async(struct pan_kmod_dev *dev,
                                     const struct pan_kmod_kbase_atom *atoms, unsigned nr_atoms,
                                     int *fence_fd, pan_kmod_kbase_done_cb done, void *data);

/* Wait until submission seq, and every one before it, has completed and run its callback. 0
 * waits for everything submitted so far. */
void pan_kmod_kbase_wait_seq(struct pan_kmod_dev *dev, uint64_t seq);

/* Wait for submission seq and every one before it, until abs_timeout_ns on CLOCK_MONOTONIC
 * (INT64_MAX: no limit). True when it has completed; seq 0 (never submitted) always has. */
bool pan_kmod_kbase_wait_seq_timeout(struct pan_kmod_dev *dev, uint64_t seq,
                                     int64_t abs_timeout_ns);

/* Submit atoms and wait for them. Returns false if the kernel refused the submission or an atom
 * completed with a fault. */
bool pan_kmod_kbase_submit(struct pan_kmod_dev *dev,
                           const struct pan_kmod_kbase_atom *atoms, unsigned nr_atoms);

/* Sync objects (see pan_kmod_kbase_sync_provider) whose signal operation has been submitted: they
 * satisfy WAIT_AVAILABLE until the completion signals them. attach_fd gives the ones still
 * submitted a sync file that signals with the GPU work, for sync file exports. */
void pan_kmod_kbase_sync_submitted(const uint32_t *handles, unsigned count);
void pan_kmod_kbase_sync_attach_fd(const uint32_t *handles, unsigned count, int fd);
/* One sync file for all of handles: -1 when all are signaled, -2 when one of them has no sync
 * file to stand for it (not submitted yet, or submitted without one). */
int pan_kmod_kbase_sync_merged_fd(const uint32_t *handles, unsigned count);
/* Records that handles are signaled by submission seq of a CSF group. */
void pan_kmod_kbase_sync_attach_csf(const uint32_t *handles, unsigned count, const void *group,
                                    uint64_t seq);
/* True when every handle is signaled or pending on a submission of group; *seq is the latest of
 * those (0 for none), which a submission on the group can wait for on the GPU. */
bool pan_kmod_kbase_sync_csf_wait_seq(const uint32_t *handles, unsigned count, const void *group,
                                      uint64_t *seq);

/* Maps the whole BO count times back to back (MEM_ALIAS) at a VA aligned to align, which is
 * both its GPU and CPU address (CPU read-only). 0 on failure. */
uint64_t pan_kmod_kbase_bo_map_repeated(struct pan_kmod_bo *bo, unsigned count, uint64_t align);
void pan_kmod_kbase_unmap_repeated(struct pan_kmod_bo *bo, uint64_t va, unsigned count);

/* ---- CSF (Valhall 2nd gen and later: command stream frontend, kbase UAPI 1.x) -------------- */

struct drm_panthor_csif_info;

/* True when the context speaks the CSF interface (queue groups and rings, no job atoms). */
bool pan_kmod_kbase_is_csf(const struct pan_kmod_dev *dev);

/* The firmware's interface, in the shape panthor reports it, so the CS builders size themselves
 * the same way on both kernels. */
const struct drm_panthor_csif_info *pan_kmod_kbase_csif_props(const struct pan_kmod_dev *dev);

#define PAN_KMOD_KBASE_CSF_MAX_QUEUES 4

/* kbase queue group priorities (BASE_QUEUE_GROUP_PRIORITY_*). */
#define PAN_KMOD_KBASE_CSF_PRIORITY_HIGH     0
#define PAN_KMOD_KBASE_CSF_PRIORITY_MEDIUM   1
#define PAN_KMOD_KBASE_CSF_PRIORITY_LOW      2
#define PAN_KMOD_KBASE_CSF_PRIORITY_REALTIME 3

struct pan_kmod_kbase_csf_group;

/* A pan_kmod_bo_alloc flag only the kbase backend knows: the BO holds sync objects a command
 * stream waits on (SYNC_WAIT). On a CSF context it becomes CSF_EVENT memory. A group whose
 * queues all wait is idle, and kbase suspends it; it is resumed only when the kernel re-reads
 * the awaited object, which it can do for CSF_EVENT memory alone. On any other memory the queue
 * group stays suspended for good (measured: all three subqueues of a PanVK queue stuck in
 * their streams, intermittently, depending on when the scheduler looked). Not for anything
 * else: kbase maps CSF_EVENT memory GPU-uncached here, and the tiler does not work out of it. */
#define PAN_KMOD_BO_FLAG_KBASE_SYNC (1u << 31)

struct pan_kmod_kbase_csf_group *
pan_kmod_kbase_csf_group_create(struct pan_kmod_dev *dev, unsigned nr_queues,
                                uint64_t compute_mask, uint64_t fragment_mask,
                                uint8_t max_compute, uint8_t max_fragment, uint8_t priority);
void pan_kmod_kbase_csf_group_destroy(struct pan_kmod_kbase_csf_group *group);

bool pan_kmod_kbase_csf_tiler_heap_create(struct pan_kmod_dev *dev, uint32_t chunk_size,
                                          uint32_t initial_chunks, uint32_t max_chunks,
                                          uint16_t target_in_flight, uint64_t *heap_ctx_va,
                                          uint64_t *first_chunk_va);
void pan_kmod_kbase_csf_tiler_heap_destroy(struct pan_kmod_dev *dev, uint64_t heap_ctx_va);

/* A command stream to CALL from a queue's ring. size == 0 only marks the queue as part of the
 * submission, so it signals with it. */
struct pan_kmod_kbase_cs_call {
   uint32_t queue;
   uint32_t size;
   uint64_t addr;
};

/* Writes the calls into the rings in order, then a completion signal on every queue that has a
 * call or is in signal_queue_mask, and kicks them. done runs on the kmod's event thread once all
 * of those queues are past it (ok = false if the group faulted). Returns the submission's
 * sequence number, 0 if nothing was submitted (done has not run then). */
uint64_t pan_kmod_kbase_csf_submit(struct pan_kmod_kbase_csf_group *group,
                                   const struct pan_kmod_kbase_cs_call *calls, unsigned nr_calls,
                                   uint32_t signal_queue_mask, uint64_t wait_seq,
                                   pan_kmod_kbase_done_cb done, void *data);

/* Waits for the submission seq (0: the latest) to complete. False on timeout. */
bool pan_kmod_kbase_csf_wait(struct pan_kmod_kbase_csf_group *group, uint64_t seq,
                             int64_t abs_timeout_ns);

/* Whether kbase reported a fatal error on the group. */
/* A sync file signaling when submission `seq` is done, or -1 when it is done already (or none
 * could be made: the caller then waits on the CPU as before). */
int pan_kmod_kbase_csf_fence_fd(struct pan_kmod_kbase_csf_group *group, uint64_t seq);
bool pan_kmod_kbase_csf_group_faulted(const struct pan_kmod_kbase_csf_group *group);

#endif /* KBASE_KMOD_H */
