target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInputCurrentDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ArrayInputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ArrayOutputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CdSocket.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ClientSocket.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CommunicationDevice.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ConnectionException.h
        ${CMAKE_CURRENT_SOURCE_DIR}/FileInputCurrentDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/FileInputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/FileOutputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/FileOutputWeightDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/InputBooleanArrayDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/InputCurrentDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/InputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/OutputBooleanArrayDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/OutputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/OutputWeightDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ServerSocket.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TCPIPConnectionType.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TCPIPInputOutputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TCPIPInputSpikeDriver.h
        ${CMAKE_CURRENT_SOURCE_DIR}/TCPIPOutputSpikeDriver.h
        )
