# Data and leakage audit

The prediction question is whether a booking will eventually be canceled. For this provisional analysis, assume a booking-record snapshot before the cancellation outcome is known. The professor must confirm the exact snapshot timing. A competition column being provided does not establish that it exists at initial booking time.

| Columns | Timing assessment and action |
|---|---|
| id | Identifier only; excluded from models. It may encode source order. Never join external hotel data to recover outcomes. |
| hotel, lead_time, arrival date fields, booked stay lengths, guests, meal, country, market_segment, distribution_channel, reserved_room_type, agent, company, customer_type, adr | Plausible booking-time fields, subject to documentation. Room, price and customer details may be updated later. Country is a customer characteristic with fairness/privacy implications. |
| is_repeated_guest, previous_cancellations, previous_bookings_not_canceled | Usable only if calculated using history strictly BEFORE this booking. Source provenance is not established by these CSVs. |
| deposit_type | Plausible booking contract information but may change later; an unusually high cancellation association is not proof of leakage or causality. Confirm definitions and timestamp. |
| booking_changes | Number of later amendments may be unavailable at initial booking; exclude in conservative sensitivity model. |
| days_in_waiting_list | Could accumulate after booking; exclude in conservative sensitivity model. |
| required_car_parking_spaces, total_of_special_requests | May be requested at booking or updated later; exclude in conservative sensitivity model. |
| is_canceled | Training target only. Never an input feature. |

The excluded outcome fields often seen in public hotel datasets, such as reservation_status and reservation_status_date, are not present here. assigned_room_type is also absent. Do not add external versions of these fields.

Learned median imputation, scaling, category dictionaries and rare-level grouping are fitted only on each training fold. Agent and company are category identifiers; missing values form an explicit category and rare categories are grouped based on training-fold counts. Unknown categories are safely handled. HistGradientBoosting receives categorical codes marked as categorical, rather than assuming their numeric codes are quantities. No full-data target encoding is used.

Exact duplicate predictor profiles can cross random folds. They may be genuine similar bookings; we preserve them in the required baseline and add a grouped-fold sensitivity check. Feature-profile matches between train and test do not reveal hidden test labels, and we do not perform label lookup. These groups cannot establish guest-level independence because no guest identifier is provided. Random CV does not establish future-time performance; a later chronological evaluation would address a different deployment question.

The main competition candidate uses all supplied non-ID predictors as a provisional record-snapshot assumption. The conservative ablation helps assess sensitivity, but does not establish deployment readiness or fix undocumented temporal history. Do not deploy without confirming timestamps and field availability.

Country/deposit tables are TRAINING descriptive statistics, not held-out effect estimates. Always read counts alongside rates; small categories have uncertain rates. Ablation score changes measure model reliance under this validation design, not what would happen if a hotel changed its policies or a guest changed country. Predictions are ranking estimates, not evidence for denying service to any nationality.
