from="dynamics"   #"dynamics"
to="meta_policy"  #"meta_policy"
out="./offlinerl/out_mb"
test_mode="False"
task=${1}
level_type=${2}
param_type=${3}
commander=${4}
exp_name="mbsac_offline_meta"

all_tasks=($task)

export PYTHONPATH=.:$PYTHONPATH

workflow=( "dynamics" "meta_policy")
echo "config"
function launch_one_exp() {
    task=$1
    from=$2
    to=$3
    local idx=0
    for entry in ${workflow[@]}
    do
        if [ "$entry" = "$from" ]; then
            from_idx=$idx
        fi
        if [ "$entry" = "$to" ]; then
            to_idx=$idx
        fi
        idx=$(( $idx + 1 ))
    done
    base_cmd="python3 examples/train_mbsac_offline_meta.py --algo_name offline_mb --task ${task} --test_mode ${test_mode} --param_type ${param_type} --level_type ${level_type} --from ${from} --to ${to} ${commander}"
    for entry in ${workflow[@]::$from_idx}
    do

        if [ ! -d "${out}/${entry}/${task}-${level_type}-${param_type}/" ]; then
          mkdir -p "${out}/${entry}/${task}-${level_type}-${param_type}/"
        fi

        base_cmd=$base_cmd" --${entry}_path ${out}/${entry}/${task}-${level_type}-${param_type}/"
        # echo $base_cmd
    done
    for entry in ${workflow[@]:$from_idx:$(( $to_idx - $from_idx + 1 ))}
    do
        if [ ! -d "${out}/${entry}/${task}-${level_type}-${param_type}/" ]; then
          mkdir -p "${out}/${entry}/${task}-${level_type}-${param_type}/"
        fi
        base_cmd=$base_cmd" --${entry}_path ${out}/${entry}/${task}-${level_type}-${param_type}/"
        # echo $base_cmd
    done
    echo $base_cmd
}

if [ ! -e ${out}/logs ]; then
    mkdir -p ${out}/logs
fi

export D4RL_SUPPRESS_IMPORT_ERROR=1

if [ ! -f "${out}/logs/${info}.stdout" ]; then
  touch "${out}/logs/${info}.stdout"
fi


for task in ${all_tasks[@]}
do
    cmd=`launch_one_exp $task $from $to`
    echo $cmd
    info="${exp_name}-${task}-${level_type}-${param_type}"
    echo $cmd --stop_point $to --exp_name "$info" "> ${out}/logs/${info}.stdout 2>&1"
    echo $cmd --stop_point $to --exp_name "$info" > "${out}/logs/${info}.stdout" 2>&1
    $cmd --stop_point $to --exp_name "$info"
done
