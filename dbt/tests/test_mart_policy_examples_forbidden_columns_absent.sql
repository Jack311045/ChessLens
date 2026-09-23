{% set mart_relation = ref('mart_policy_examples') %}

select
    column_name
from information_schema.columns
where table_schema = '{{ mart_relation.schema }}'
  and table_name = '{{ mart_relation.identifier }}'
  and lower(column_name) in (
      'result',
      'final_result_label',
      'termination',
      'termination_label',
      'ply_count',
      'game_ply_count'
  )
