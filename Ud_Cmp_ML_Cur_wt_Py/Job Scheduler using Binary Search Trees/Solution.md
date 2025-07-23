## Our proposed solution...
- Based on these specs, the architects/designers/developers (you and your team) have decided that the Binary Search Tree structure fits these specifications perfectly.
- The key for our nodes will be time. Since time is continuous and we can only have one job run at a given time, the BST will not allow for duplicate jobs and be easy to determine the scheduling.
- Scheduling can happen at any time for later times in the day, so the order in which jobs are scheduled (insertions) will be random, but based on the BST structure we'll always be able to get a sorted view of our data based on the ordering of nodes that a BST allows maintains.
- We can traverse the BST easily using in-order traversal to get a sorted view of our daily scheduled jobs.
- We can easily add restrictions on the insertion of jobs to the schedule to account for no overlap.
- Insertion, removal and other operations like number of jobs, finding a specific job (not a requirement at this time) are very fast in a BST.
