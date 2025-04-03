import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import shutil
from geopy.distance import geodesic
import math
from scipy.interpolate import interp1d
from scipy.signal import resample
import shutil

def delete_all_files_and_folders(delete_path):
    """폴더 내 모든 파일 및 폴더 삭제 함수"""
    if not os.path.exists(delete_path):  # 폴더가 존재하는지 확인
        print("폴더가 존재하지 않습니다:", delete_path)
        return

    for item_name in os.listdir(delete_path):  # 폴더 내 모든 항목 가져오기
        item_path = os.path.join(delete_path, item_name)

        if os.path.isfile(item_path):  # 파일이면 삭제
            os.remove(item_path)
            print(f"파일 삭제됨: {item_path}")

        elif os.path.isdir(item_path):  # 폴더이면 삭제 (하위 내용 포함)
            shutil.rmtree(item_path)
            print(f"폴더 삭제됨: {item_path}")


def calculate_distance(row, df):
    if pd.notnull(row["Latitude"]) and row.name > 0:  # 첫 번째 행은 이전 값이 없으므로 예외 처리
        prev_row = df.iloc[row.name - 1]  # 이전 행 가져오기
        return geodesic((prev_row["Latitude"], prev_row["Longitude"]),
                        (row["Latitude"], row["Longitude"])).meters

validation_path="D:\\SHL-2023-Validate\\validate"

def calculate_gps_features(df):
    """
    GPS 데이터에서 속도, 가속도, 저크, 방향 변화율(Bearing Rate) 계산
    """
    # 타임스탬프 변환
    df["Timestamp_init"]=df["Timestamp"]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    
    # 시간 차이 (초 단위)
    df["Time_diff"] = df["Timestamp"].diff().dt.total_seconds()
    df["Delta_Latitude"]=df["Latitude"].diff().fillna(0)
    df["Delta_Longitude"]=df["Longitude"].diff().fillna(0)
    # 거리 계산 (위도, 경도 사용)
    df["Distance"] = df.apply(lambda row: calculate_distance(row, df), axis=1)
    # 속도 계산 (m/s)
    df["Speed"] = df["Distance"] / df["Time_diff"]
    
    # 가속도 계산 (m/s²)
    df["Speed_diff"] = df["Speed"].diff()
    df["Acceleration"] = df["Speed_diff"] / df["Time_diff"]
    
    # 저크 계산 (m/s³)
    df["Acceleration_diff"] = df["Acceleration"].diff()
    df["Jerk"] = df["Acceleration_diff"] / df["Time_diff"]
    
    # Bearing(방향) 계산 함수
    def calculate_initial_compass_bearing(lat1, lon1, lat2, lon2):
        if pd.isnull(lat1) or pd.isnull(lon1) or pd.isnull(lat2) or pd.isnull(lon2):
            return np.nan

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        diff_lon = lon2 - lon1

        x = math.sin(diff_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon))

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        return (initial_bearing + 360) % 360  # 0~360 범위로 변환

    # Bearing(방향) 계산
    df["Bearing"] = np.vectorize(calculate_initial_compass_bearing)(
        df["Latitude"].shift(), df["Longitude"].shift(), df["Latitude"], df["Longitude"]
)

    # Bearing 변화율 계산 (deg/s)
    df["Bearing_diff"] = df["Bearing"].diff()
    df["Bearing_rate"] = df["Bearing_diff"] / df["Time_diff"]
    

    return df

def assign_activity(timestamp, labels_df):
    condition = (labels_df["Start_time"] <= timestamp) & (timestamp <= labels_df["End_time"])
    matched_activity = labels_df.loc[condition, "Label_main"]
    return matched_activity.iloc[0] if not matched_activity.empty else "Unknown"


Acc_columns=["Timestamp",
               "Acceleration_X","Acceleration_Y","Acceleration_Z"]
Gyr_columns=[ "Timestamp", "Gyroscope_X","Gyroscope_Y","Gyroscope_Z"]
Mag_columns=[ "Timestamp", "Magnetometer_X","Magnetometer_Y","Magnetometer_Z" ]
Motion_columns=["Timestamp",
               "Acceleration_X","Acceleration_Y","Acceleration_Z",
               "Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
               "Magnetometer_X","Magnetometer_Y","Magnetometer_Z" ]

GPS_columns=["Timestamp","Accuracy","Latitude","Longitude","Altitude"]
Label_main_df_columns=["Start_time","End_time","Label_main"]
Motion_df=pd.DataFrame()
GPS_df=pd.DataFrame()
Label_df=pd.DataFrame()
motion_file=""
gps_file=""

Label_df=pd.DataFrame()
Acc_df=pd.DataFrame()
Gyr_df=pd.DataFrame()
Mag_df=pd.DataFrame()
def resample_Motion_data(motion_df, motion_freq=100):
    """
    모션 데이터 (가속도계, 자이로, 자기장 등)를 100Hz로 리샘플링하는 함수.
    """
    df=pd.DataFrame()
    df["Label"]=motion_df["Label"]
    df["Timestamp"]=motion_df["Timestamp"].astype(float)
    motion_df["Timestamp"] = pd.to_datetime(motion_df["Timestamp"], unit="ms").astype("int64") / 1e9
    min_time, max_time = motion_df["Timestamp"].min(), motion_df["Timestamp"].max()
    new_time_motion = np.linspace(min_time, max_time, int((max_time - min_time) * motion_freq))


    # 보간을 위한 딕셔너리 생성
    resampled_motion = {}

    for col in Motion_columns:
        if col in motion_df.columns:  # 해당 센서 데이터가 존재하는지 확인
            interp_func = interp1d(motion_df["Timestamp"], motion_df[col], kind='linear', fill_value="extrapolate")
            resampled_motion[col] = interp_func(new_time_motion)

    # 데이터프레임 변환
    motion_resampled_df = pd.DataFrame(resampled_motion)
    motion_resampled_df["Timestamp"] = new_time_motion  # 시간 정보 추가


    df["Timestamp"] = pd.to_numeric(df["Timestamp"])
    motion_resampled_df["Timestamp"] = pd.to_numeric(motion_resampled_df["Timestamp"])

    motion_resampled_df = pd.merge_asof(motion_resampled_df, df, on="Timestamp", direction="nearest")
    return motion_resampled_df


Acc_df=pd.DataFrame()
Gyr_df=pd.DataFrame()
Mag_df=pd.DataFrame()
Label_df=pd.DataFrame()
Motion_df=pd.DataFrame()
an_folder_dir="D:\\SHL\\an_pre"
delete_all_files_and_folders(an_folder_dir)
for folders in os.listdir(validation_path):
    file_path=validation_path+"\\"+folders
    Acc_df=pd.DataFrame()
    Gyr_df=pd.DataFrame()
    Mag_df=pd.DataFrame()
    Label_df=pd.DataFrame()
    Motion_df=pd.DataFrame()
    for files in os.listdir(file_path):    
        
            
        if "Acc" in files:
    # 데이터 읽기 (공백을 구분자로 사용)
              Acc_file=files
              Acc_df= pd.read_csv(file_path+"\\"+Acc_file, delim_whitespace=True, header=None)
              # Acc_df=Acc_df.iloc[:,:10]
              Acc_df.columns=Acc_columns #acceleration,gyroscope,magnetometer

        elif "Gyr" in files:
    # 데이터 읽기 (공백을 구분자로 사용)
              Gyr_file=files
              Gyr_df= pd.read_csv(file_path+"\\"+Gyr_file, delim_whitespace=True, header=None)
              # Gyr_df=Gyr_df.iloc[:,:10]
              Gyr_df.columns=Gyr_columns #acceleration,gyroscope,magnetometer

        elif "Mag" in files:
    # 데이터 읽기 (공백을 구분자로 사용)
              Mag_file=files
              Mag_df= pd.read_csv(file_path+"\\"+Mag_file, delim_whitespace=True, header=None)
              # Mag_df=Mag_df.iloc[:,:10]
              Mag_df.columns=Mag_columns #acceleration,gyroscope,magnetometer
    
        # elif "Location" in files:
        #     gps_file=files
        #     GPS_df=pd.read_csv(file_path+"\\"+files, sep=r"\s+", header=None)
            
        #                   axis=1)
        #     GPS_df.columns=GPS_columns

            # GPS_df=calculate_gps_features(GPS_df)
    
        elif "Label" in files:
            Label_files=files
            Label_df=pd.read_csv(file_path+"\\"+Label_files, sep=r"\s+", header=None)
            Label_df=Label_df.rename(columns={0:"Timestamp",1:"Label"})

 
    Label_df = Label_df.sort_values("Timestamp")
    
    Acc_df=Acc_df.sort_values("Timestamp")
    Gyr_df= Gyr_df.sort_values("Timestamp")
    Mag_df=Mag_df.sort_values("Timestamp")
    
    
    Label_df = Label_df.drop_duplicates(subset=['Timestamp'])
                # # 중복된 Timestamp만 추출
                #     duplicate_timestamps = Label_df[Label_df.duplicated(subset=['Timestamp'], keep=False)]
                
                # # 어떤 값이 몇 번 중복됐는지 확인
                #     dup_count = Label_df['Timestamp'].value_counts()
                #     dup_count = dup_count[dup_count > 1]
                
                #     # Acc_df에 Label 정보 병합
    # label_df에 있는 Timestamp만 사용
    Acc_df= Acc_df[Acc_df['Timestamp'].isin(Label_df['Timestamp'])]
    Gyr_df = Gyr_df[Gyr_df['Timestamp'].isin(Label_df['Timestamp'])]
    Mag_df = Mag_df[Mag_df['Timestamp'].isin(Label_df['Timestamp'])]

    Acc_df= pd.merge(Acc_df, Label_df, on="Timestamp", how="inner")       
    Gyr_df = pd.merge(Gyr_df, Label_df, on="Timestamp", how="left")
    Mag_df = pd.merge(Mag_df, Label_df, on="Timestamp", how="left")
        

    
    Label_df = Label_df.sort_values("Timestamp")
    # GPS_df["Timestamp"]=GPS_df["Timestamp_init"]
    # GPS_df=GPS_df.drop(columns=["Timestamp_init"])
    # GPS_df=GPS_df.sort_values("Timestamp")

    #📌`merge_asof()`를 사용하여 가장 가까운 Label 매칭
    # GPS_df = pd.merge_asof(GPS_df, Label_df.iloc[:,:2], on="Timestamp", direction="nearest")
    # GPS_df.to_csv(an_folder_dir+"\\"+folders+"_validation_"+gps_file.replace("Location.txt","Gps")+".csv",index=False)
    
    # Acc_df.to_csv(an_folder_dir+"\\"+folders+"_validation_"+Acc_file.replace(".txt","")+".csv",index=False)
    # Mag_df.to_csv(an_folder_dir+"\\"+folders+"_validation_"+Mag_file.replace(".txt","")+".csv",index=False)
    # Gyr_df.to_csv(an_folder_dir+"\\"+folders+"_validation_"+Gyr_file.replace(".txt","")+".csv",index=False)
    Motion_df=pd.concat([Acc_df,Gyr_df.iloc[:,1:],Mag_df.iloc[:,1:],Label_df.iloc[:,1]],axis=1)

    Motion_df = pd.merge( Acc_df,  Gyr_df, on="Timestamp", how="inner")
    Motion_df=pd.merge( Motion_df, Mag_df, on="Timestamp", how="inner")
    # Motion_df.to_csv(an_folder_dir+"\\"+folders+"_validation_Motion.csv",index=False)
# Motion_df=pd.DataFrame()


acc_timestamps = set(Acc_df['Timestamp'])
gyr_timestamps = set(Gyr_df['Timestamp'])
mag_timestamps = set(Mag_df['Timestamp'])

label_timestamps = set(Label_df['Timestamp'])

common_ts = acc_timestamps & label_timestamps
print(f"공통 Timestamp 수: {len(common_ts)}")

# Acc_df에는 있는데, label_df에는 없는 값
only_in_acc = acc_timestamps - mag_timestamps

# label_df에는 있는데, Acc_df에는 없는 값
only_in_gyr =mag_timestamps-acc_timestamps

print(f"Acc_df에만 있는 Timestamp 수: {len(only_in_acc)}")
print(f"label_df에만 있는 Timestamp 수: {len(only_in_gyr)}")


#########################################################################################################################

Motion_columns=["Timestamp",
               "Acceleration_X","Acceleration_Y","Acceleration_Z",
               "Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
               "Magnetometer_X","Magnetometer_Y","Magnetometer_Z" ]

GPS_columns=["Timestamp","Accuracy","Latitude","Longitude","Altitude"]

motion_df=pd.DataFrame()
motion_bag=pd.DataFrame()
motion_hand=pd.DataFrame()
motion_hips=pd.DataFrame()
motion_torso=pd.DataFrame()
gps_bag=pd.DataFrame()
gps_hand=pd.DataFrame()
gps_hips=pd.DataFrame()
gps_torso=pd.DataFrame()

for f in os.listdir(an_folder_dir):
    if "Motion" in f:
        # if "Bag" in f:
        #     motion_bag=pd.read_csv(os.path.join(an_folder_dir,f))
        #     motion_bag=motion_bag.drop(columns=["Label_x","Label_y"])
        #     motion_bag = resample_Motion_data(motion_bag)

        if "Hand" in f:
            motion_hand=pd.read_csv(os.path.join(an_folder_dir,f))
        # elif "Hips" in f:
        #     motion_hips=pd.read_csv(os.path.join(an_folder_dir,f))
        # elif "Torso" in f:
        #     motion_torso=pd.read_csv(os.path.join(an_folder_dir,f))
        
    # else :
    #     if "Bag" in f:
    #         gps_bag=pd.read_csv(os.path.join(an_folder_dir,f))
    #         gps_resampled = resample_gps_data(gps_bag)  # 1Hz 리샘플링
    #         gps_resampled=calculate_gps_features(gps_resampled)
    #         gps_resampled["Timestamp"]=gps_resampled["Timestamp_init"]
    #         gps_resampled=gps_resampled.drop(columns=["Timestamp_init"])
    #         gps_resampled=gps_resampled.sort_values("Timestamp")
    #     elif "Hand" in f:
    #         gps_hand=pd.read_csv(os.path.join(an_folder_dir,f))
    #     elif "Hips" in f:
    #         gps_hips=pd.read_csv(os.path.join(an_folder_dir,f))
    #     elif "Torso" in f:
    #         gps_torso=pd.read_csv(os.path.join(an_folder_dir,f))
        

# motion_bag= np.concatenate(motion_bag, axis=0)
# motion_hand = np.concatenate(motion_hand, axis=0)
# motion_hips= np.concatenate(motion_hips, axis=0)
# motion_torso= np.concatenate(motion_torso, axis=0)

        # Motion_columns=["Timestamp",
        #                "Acceleration_X","Acceleration_Y","Acceleration_Z",
        #                "Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
        #                "Magnetometer_X","Magnetometer_Y","Magnetometer_Z" ,
        #                "Label"]
    
        # GPS_columns=["Timestamp","Accuracy","Latitude","Longitude","Altitude"]
    
        # motion_bag = pd.DataFrame(motion_bag, columns=Motion_columns)
        # motion_hand = pd.DataFrame(motion_hand, columns=Motion_columns)
        # motion_hips = pd.DataFrame(motion_hips, columns=Motion_columns)
        # motion_torso = pd.DataFrame(motion_torso, columns=Motion_columns)

# motion_bag = motion_bag.astype(np.float32)
# motion_hand = motion_hand.astype(np.float32)
# motion_hips = motion_hips.astype(np.float32)
# motion_torso = motion_torso.astype(np.float32)
# _________________________________________________________________________________________________________________________________________
def segment_trips(data):
    """
    타임스탬프 간격이 10ms 이상 발생할 경우 새로운 Trip으로 분할하는 함수
    """
    # 타임스탬프 차이 계산 (ms 단위)
    data["Time_diff"] = data["Timestamp"].diff()

    # 10ms 이상 간격이 발생하면 새로운 Trip 시작
    data["Trip_id"] = ((data["Time_diff"] > 10) | (data["Label"] != data["Label"].shift())).cumsum()

  

    # 필요 없는 열 제거
    data = data.drop(columns=["Time_diff"])

    return data



def sliding_window_to_tensor(df, window_size, freq, to_tensor=True):
    """
    슬라이딩 윈도우 적용 후 3D NumPy 배열 또는 PyTorch 텐서로 변환

    Parameters:
        df (list, np.ndarray, pd.DataFrame): 입력 데이터
        window_size (int): 윈도우 크기 (초 단위)
        overlap (float): 윈도우 오버랩 비율 (0 ~ 1)
        freq (int): 샘플링 주파수 (Hz)
        to_tensor (bool): PyTorch Tensor로 변환 여부 (True: Tensor, False: NumPy 배열)

    Returns:
        3D NumPy 배열 또는 PyTorch Tensor (batch_size, channels, time_steps)
    """
    stride=100
    # step_size = int(window_size * freq * (1 - overlap))  # 윈도우 이동 간격
    samples_per_window = window_size * freq  # 한 윈도우 내 샘플 개수
    segments = []

    for start in range(0, len(df) - samples_per_window + 1, samples_per_window):
        # (channels, time_steps)
        segment = np.array(
            df[start:start + samples_per_window], dtype=np.float32).T

        segments.append(segment)

    # ✅ 3차원 변환 (Batch, Channels, Time)
    tensor_3d = np.array(segments)  # (batch_size, channels, time_steps)

    # ✅ PyTorch Tensor 변환
    if to_tensor:
        return torch.tensor(tensor_3d, dtype=torch.float32)

    return tensor_3d 

                
                # delete_all_files_and_folders("D:\\SHL\\an_pre\\hand_val_seg")
                # output_folder_path="D:\\SHL\\an_pre\\hand_val_seg"
                # motion_hand=motion_hand.drop(columns=["Label_x","Label_y"])
seg_Motion_df=segment_trips(motion_hand)
                
                
                #                 # # Trip별로 파일 저장
                #                 # for trip_id, trip_data in seg_Motion_df.groupby("Trip_id"): 
                #                 #     file_path = output_folder_path+f"_{trip_id}_Label_{trip_data['Label'].unique()}.csv"
                   
                #     trip_data.to_csv(file_path+".csv", index=False)
                #     print(f"Trip {trip_id} 저장 완료: {file_path}")
                    
                        

# motion_bag = sliding_window_to_tensor(motion_bag, window_size=60, overlap=0.5, freq=100)
seg_hand = sliding_window_to_tensor(seg_Motion_df, window_size=60, freq=100)
# motion_hips = sliding_window_to_tensor(motion_hips, window_size=60, overlap=0.5, freq=100)
# motion_torso = sliding_window_to_tensor(motion_torso, window_size=60, overlap=0.5, freq=100)


#


























