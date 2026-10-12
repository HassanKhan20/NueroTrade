' NeuroTrade reminder — fires 5 min before the daily rebalance task.
' Auto-dismisses after 60 seconds so it never blocks the user.
'
' Triggered by Windows Task Scheduler ("NeuroTrade Reminder").
Set WshShell = CreateObject("WScript.Shell")
msg = "NeuroTrade rebalance task fires in 5 minutes (2:50 PM CT)." & vbCrLf & vbCrLf & _
      "Keep laptop awake & online for ~10 seconds." & vbCrLf & _
      "After 2:55 PM CT, you can close it — orders are queued at Alpaca."
WshShell.Popup msg, 60, "NeuroTrade Reminder", 64
