#!/usr/bin/env python3
"""
Initial run script for GROMACS 2022

ややこしいことをやっているように見えるが、gromacsのコマンドを呼びだすために
必要なパラメータを揃えているだけ。

なので、"Invoke external command: {command}"で
表示されるコマンドをそのままシェルで打ちこんでも同じことができる。
"""

import os
import sys
from logging import INFO, basicConfig, getLogger

GMX = os.environ.get(
    "GMX", "gmx"
)  # 環境変数GMXが設定されていたら、それをパスとみなす。


def confirm_files(files, status):
    """
    必要なファイルが存在するかどうかを事前にチェックする。
    """
    for file in files:
        if status:
            message = f"Missing file: {file}"
        else:
            message = f"File already exists: {file}"
        assert os.path.exists(file) == status, message


def run(
    mdp="continue.mdp", job_name="4P2005", N_thread=8, job_range=(1, 2), index_file=None
):
    """
    初期配置からの最初の実行

    パラメータ
    num         ジョブの通し番号。2を指定する場合は、00001.*ファイルが必要。
    mdp         mdpファイルのパス
    job_name    ジョブの名前。topファイル、groファイルと同じ名前にする。指定しなければ'4P2005'が使われる。
    N_thread    並列計算で使用するcore数
    index_file  分子を固定する場合に使う、.ndxファイルのパス

    numが1の場合に限り、groファイルを読みこむ。それ以外は続きを計算する。

    返り値
    なし
    """
    logger = getLogger()

    for num in range(*job_range):
        # tprファイルを作成するのに必要なファイルがあることを確認する。
        if num == 1:
            gro = f"{job_name}.gro"
            top = f"{job_name}.top"
            confirm_files([gro, top], status=True)
        else:
            previous_tpr = f"{num-1:05d}.tpr"
            previous_cpt = f"{num-1:05d}.cpt"
            top = f"{job_name}.top"
            confirm_files([previous_tpr, previous_cpt], status=True)

        # 計算済みのデータを上書きするのを避けるため、書きこみ予定のファイルが存在しないことを確認する。
        tpr = f"{num:05d}.tpr"
        cpt = f"{num:05d}.cpt"
        confirm_files([tpr, cpt], status=False)

        options = dict()
        options["-maxwarn"] = 2
        options["-f"] = mdp
        options["-p"] = top
        options["-o"] = tpr
        if num == 1:
            # 初回のみ
            options["-c"] = gro
        else:
            options["-c"] = previous_tpr
            options["-t"] = previous_cpt
        if index_file is not None:
            options["-n"] = index_file

        # gromppにより、バイナリインプットファイルtprを生成
        option_str = ""
        for key, value in options.items():
            option_str += f"{key} {value} "
        command = f"{GMX} grompp {option_str} > {num:05d}.grompp.log 2>&1"
        # if num == 1:
        #     command = f"{GMX} grompp -maxwarn 2 -f {mdp} -p {top} -c {gro} -o {tpr} > {num:05d}.grompp.log 2>&1"
        # else:
        #     command = f"{GMX} grompp -maxwarn 2 -f {mdp} -p {top} -c {previous_tpr} -t {previous_cpt} -o {tpr} > {num:05d}.grompp.log 2>&1"
        logger.info(f"Invoke external command: {command}")
        err = os.system(command)
        logger.info(f"Return code: {err}")
        if err:
            logger.error(f"Terminated by an error {err}.")
            sys.exit(1)

        # mdrunによりMDを実行
        command = f"{GMX} mdrun -notunepme  -nt {N_thread} -deffnm {num:05d} -c {num:05d}-last.gro > {num:05d}.mdrun.log 2>&1"
        logger.info(f"Invoke external command: {command}")
        os.system(command)
        logger.info(f"Return code: {err}")


if __name__ == "__main__":
    logger = getLogger()
    basicConfig(level=INFO)
    cmd = sys.argv.pop(0)
    index_file = None
    if sys.argv[0] == "-n":
        sys.argv.pop(0)
        index_file = sys.argv.pop(0)
    if len(sys.argv) not in (2, 3, 4):
        logger.error(f"Usage: {sys.argv[0]} job_name mdpfile [job#]")
        logger.error("\tjob_name  Name of the .top file.")
        logger.error(
            "\tjob#      Either empty, single number, or two numbers specifying the first and last job number."
        )
        sys.exit(1)

    # job_nameとmdpファイルは必ず引数で与える。
    job_name = sys.argv.pop(0)
    mdp = sys.argv.pop(0)

    # 通し番号は引数で与える。指定しなければ1となる。
    head, tail = 1, 1
    if len(sys.argv) > 0:
        head = int(sys.argv.pop(0))
        tail = head
    if len(sys.argv) > 0:
        tail = int(sys.argv.pop(0))
    job_range = [head, tail + 1]
    run(job_name=job_name, mdp=mdp, job_range=job_range, index_file=index_file)
