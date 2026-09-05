// Polyphon Studio Application Entry Point & Bootstrap

document.addEventListener('DOMContentLoaded', () => {
  // Initial system status check & recurring poll
  checkSystemStatus();
  setInterval(checkSystemStatus, 15000);

  // Check current background processing job & poll
  checkCurrentJob();
  setInterval(checkCurrentJob, 4000);
});

