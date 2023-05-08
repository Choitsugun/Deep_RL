import argparse

def set_args():
    parser = argparse.ArgumentParser()

    # =============== dataset prep ============== #
    parser.add_argument('--max_dial_len', default=140, type=int, help='The max number of token in a dial for actor training')
    parser.add_argument('--max_cont_len', default=110, type=int, help='The max number of token in a cont for actor improven')
    parser.add_argument('--num_state',    default=3,   type=int, help='The number of utteance to keep for state, required >=1')
    parser.add_argument('--max_stac_len', default=260, type=int, help='The max number of token in a state-action pair for DQN training')

    # ============= common setting ============= #
    parser.add_argument('--log_path',            default='../save_load/log/log.log', type=str)
    parser.add_argument('--save_model_path',     default='../save_load/model',       type=str)
    parser.add_argument('--dataset_path',        default='../dataset/',              type=str)
    parser.add_argument('--ds_cache_dir',        default='../download/',             type=str)

    parser.add_argument('--DQN_pretrained',      default='../save_load/pre_weight/BERT',                type=str)
    parser.add_argument('--DQN_fgr_checkp',      default='../save_load/model/DQN/fgr/epoch200',         type=str)
    parser.add_argument('--DQN_cgr_checkp',      default='../save_load/model/DQN/cgr/epoch210',         type=str)
    parser.add_argument('--GPT2_pretrained',     default='../save_load/pre_weight/GPT2',                type=str)
    parser.add_argument('--GPT2_checkp',         default='../save_load/model/actor/{}/GPT2/epoch20',    type=str)
    parser.add_argument('--DialoGPT_pretrained', default='../save_load/pre_weight/DialoGPT',            type=str)
    parser.add_argument('--DialoGPT_checkp',     default='../save_load/model/actor/{}/DialoGPT/epoch20',type=str)
    parser.add_argument('--T5_pretrained',       default='../save_load/pre_weight/T5',                  type=str)
    parser.add_argument('--T5_checkp',           default='../save_load/model/actor/{}/T5/epoch20',      type=str)
    parser.add_argument('--GODEL_pretrained',    default='../save_load/pre_weight/GODEL',               type=str)
    parser.add_argument('--GODEL_checkp',        default='../save_load/model/actor/{}/GODEL/epoch20',   type=str)
    parser.add_argument('--Roberta_pretrained',  default='../save_load/pre_weight/Roberta',             type=str)
    parser.add_argument('--AtEc_checkp',         default='../save_load/model/AtEc/epoch57/AtEc.pth',    type=str)
    parser.add_argument('--Forimpr_pretrained',  default='../save_load/pre_weight/{}',                  type=str)
    parser.add_argument('--Forimpr_checkp',      default='../save_load/model/actor/wo_tp/{}/epoch20',   type=str)
    parser.add_argument('--Forinte_pretrained',  default='../save_load/pre_weight/{}',                  type=str)
    #parser.add_argument('--Forinte_checkp',      default='../save_load/model/actor/wo_tp/{}/epoch20',   type=str)
    parser.add_argument('--Forinte_checkp',      default='../save_load/model/actor_im/stan/{}/epoch20', type=str)
    #parser.add_argument('--Forinte_checkp',      default='../save_load/model/actor_im/ours/{}/epoch20', type=str)

    parser.add_argument('--cgr_enable', default=True,   type=bool, help='Whether to use the dual granularity for training')
    parser.add_argument('--actor_name', default="GPT2", type=str,  help='Required set as: GPT2, DialoGPT, T5, GODEL')
    parser.add_argument('--device',     default='0',    type=str)
    parser.add_argument('--batch_size', default=40,     type=int)
    parser.add_argument('--epochs',     default=200,     type=int)
    parser.add_argument('--warm_step',  default=6000,   type=int)
    parser.add_argument('--n_worker',   default=1,      type=int)
    parser.add_argument('--lr',         default=2.4e-5, type=float)

    # ========== train AtEc, DNQ, actor ========== #
    parser.add_argument('--sali_path',  default='../dataset/sali_stin/sali.txt', type=str)
    parser.add_argument('--gama',       default=0.98,  type=float, help='Affect decay rate of reward')
    parser.add_argument('--inter_sync', default=30,    type=int,   help='The interval set of sync the qtar')
    parser.add_argument('--after_save', default=195,   type=int,   help='Saving the acotr or DQN after the epoch')
    parser.add_argument('--topi_penal', default=0.15,  type=float, help='The penalty of 3: diaries and daily life')
    parser.add_argument('--ncla_topic', default=19,    type=int,   help='This is a fixed value, please do not change')
    parser.add_argument('--forwa_only', default=False, type=bool,  help='Whether only evaluation')
    parser.add_argument('--patience',   default=10,    type=int,   help='The number of patience times')
    parser.add_argument('--dime_model', default=512,   type=int,   help='The embedd size of the AtEc model')

    # ========== inference & improvement ======== #
    parser.add_argument('--imprdt_path', default='../save_load/result_imdata', type=str, help='The path of save improved data')
    parser.add_argument('--from_imprdt', default=False, type=bool, help='Whether to load the improved data from saved')
    parser.add_argument('--n_retur_seq', default=1,     type=int,  help='The number of generations, required >=1')
    parser.add_argument('--temperature', default=1.5,   type=float)
    parser.add_argument('--resp_ge_len', default=50,    type=int)

    # ========= intaraction with blender ======== #
    parser.add_argument('--blender_model',   default='../save_load/pre_weight/Blenderbot', type=str)
    parser.add_argument('--first_mess_path', default="../dataset/sali_stin/stin.txt",      type=str)
    parser.add_argument('--save_inter_path', default='../save_load/result_interac',        type=str)

    parser.add_argument('--n_turns',          default=5,    type=int)
    parser.add_argument('--n_episode',        default=100,  type=int)
    parser.add_argument('--blende_num_state', default=3,    type=int)
    parser.add_argument('--dummy_actor_mess', default='hi', type=str)

    # ================ analysis =============== #
    parser.add_argument('--resu_path', default="../save_load/result_genera", type=str, help='For make generation result')
    parser.add_argument('--anal_path', default="../save_load/result_interac/interaction-1.txt", type=str, help='For sp')
    parser.add_argument('--n_anal',    default=300,  type=int, help='For ce, ac, and generation samples from test set')
    parser.add_argument('--gene_resu', default=True, type=bool, help='Whether output the generation result')

    return parser.parse_args()