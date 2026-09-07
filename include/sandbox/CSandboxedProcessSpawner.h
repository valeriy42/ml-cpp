/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#ifndef INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
#define INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h

#include <core/CProcess.h>
#include <sandbox/ImportExport.h>

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace ml {
namespace sandbox {

//! \brief
//! Spawn processes inside a Sandbox2 isolation boundary.
//!
//! DESCRIPTION:\n
//! Used by the ML controller to launch pytorch_inference with filesystem and
//! syscall restrictions. PID lifetime is tracked via Sandbox2 AwaitResult
//! monitor threads rather than waitpid(), because the sandboxee is a child of
//! the Sandbox2 forkserver rather than of the controller.
//!
class SANDBOX_EXPORT CSandboxedProcessSpawner {
public:
    using TStrVec = std::vector<std::string>;

public:
    CSandboxedProcessSpawner();
    ~CSandboxedProcessSpawner();

    //! Spawn a sandboxed process. Returns true on success.
    bool spawn(const std::string& processPath,
               const TStrVec& args,
               core::CProcess::TPid& childPid,
               std::string* failureReason = nullptr);

    //! Kill a sandboxed child process started by this object.
    bool terminateChild(core::CProcess::TPid pid);

    //! Returns true if this object spawned a sandboxed process with the given
    //! PID that is still running.
    bool hasChild(core::CProcess::TPid pid) const;

private:
    using TPidSet = std::set<core::CProcess::TPid>;

    //! \brief The set of live sandboxed PIDs, and the lock that guards it.
    //!
    //! DESCRIPTION:\n
    //! Held behind a shared_ptr because the monitor thread that removes a PID
    //! outlives the spawn() call that started it, and can outlive this object:
    //! the controller may tear the spawner down while a sandboxed
    //! pytorch_inference is still running, and the monitor only learns that the
    //! sandboxee exited some time later. A raw pointer back to the spawner
    //! would be dangling by then, so the monitor co-owns the registry instead
    //! and the spawner needs no synchronisation in its destructor.
    struct SPidRegistry {
        mutable std::mutex s_Mutex;
        TPidSet s_Pids;
    };
    using TPidRegistryPtr = std::shared_ptr<SPidRegistry>;

    const TPidRegistryPtr m_PidRegistry{std::make_shared<SPidRegistry>()};
};

} // namespace sandbox
} // namespace ml

#endif // INCLUDED_ml_sandbox_CSandboxedProcessSpawner_h
