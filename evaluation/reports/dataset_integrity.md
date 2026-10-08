# Dataset integrity report

- Total cases: 14
- Real cases: 6
- Synthetic cases: 8
- Extreme-tagged cases: 3
- Missing GT: 5
- Placeholder-named: 2
- Duplicate image groups: 4
- Duplicate GT groups: 1
- Cross-split leaks: 4
- Qualification eligible count: 0
- Qualification eligible: **False**
- Decision: `NOT_READY_DATASET`

## Splits

- `dev`: {"exists": true, "n": 4, "real": 0, "synthetic": 4, "with_gt": 4}
- `held_out`: {"exists": true, "n": 10, "real": 6, "synthetic": 4, "with_gt": 5}
- `qualification`: {"exists": true, "n": 0, "real": 0, "synthetic": 0, "with_gt": 0}

## Qualification failures

- fewer_than_15_real_qualification_pages (have 0; target 30)

## Duplicate images

- dev:messy_rx_sample, held_out:messy_rx_sample
- dev:messy_note, held_out:messy_note
- dev:placeholder_messy_note, held_out:placeholder_messy_note
- dev:printed_report_sample, held_out:printed_report_sample

## Cross-split leaks

- {'sha256': '50b6dd74b295d510c1d1bceef1bdd8afedef3a2355fc57d298353415bae2433a', 'cases': ['dev:messy_rx_sample', 'held_out:messy_rx_sample'], 'splits': ['dev', 'held_out']}
- {'sha256': 'cf9be8eaf829fdba474c6ee1653e5a33e5958579ac36fde57bd671bd4906e55c', 'cases': ['dev:messy_note', 'held_out:messy_note'], 'splits': ['dev', 'held_out']}
- {'sha256': '1b8c303d998b9390c4b1a64ae4b8b3cd37dd7d5fe871e062771abfeea43d0bac', 'cases': ['dev:placeholder_messy_note', 'held_out:placeholder_messy_note'], 'splits': ['dev', 'held_out']}
- {'sha256': '83a8a69e2009abeffc8705a86aa34a6663cbf86ce433d12cf31d956c4de0256e', 'cases': ['dev:printed_report_sample', 'held_out:printed_report_sample'], 'splits': ['dev', 'held_out']}
